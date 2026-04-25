import torch
import math
import time
import sys
import random
import numpy as np
from .preprocess import * 
from .model_utils import compute_grad_norm
from .model import *
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric import seed_everything
from copy import deepcopy
import matplotlib.pyplot as plt

class Model():

    def __init__(self, 
        adata,
        slice, 
        data = "ST",
        gnn_layer = 'GAT',
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_t=5e-5,
        weight_decay=0.00,
        epochs=600, 
        neighbors=3,
        radius_n = 150, 
        dim_input=3000,
        dim_output=32,
        random_seed = 41,
        adj_reconstruction = False,
        metric = 'cosine',
        annealing = 'linear', 
        ft_epochs = 0,
        graph_loss = False, 
        init_t = 6, 
        schedule_t = False, 
        dtype= torch.float64, 
        ST = 'Visium',
        verbose = False, 
        k_iter = 30,
        n_iter = 100,
        scaling = True,
        drop_housekeeping = True
        ):


        self.adata = adata.copy()
        self.data = data
        self.gnn_layer = gnn_layer
        self.device = device
        self.learning_rate=learning_rate
        self.learning_rate_t = learning_rate_t
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.adj_reconstruction = adj_reconstruction
        self.annealing = annealing
        self.ft_epochs = ft_epochs
        self.graph_loss = graph_loss
        self.init_t = init_t
        self.slice = slice
        self.schedule_t = schedule_t
        self.dtype = dtype
        self.t = None
        self.verbose = verbose
        self.ST = ST
        self.k_iter = k_iter
        self.n_iter = n_iter
        self.scaling = scaling
        self.drop_housekeeping = drop_housekeeping

        if self.annealing in ['cosine', 'linear']:
            self.tau_start, self.tau_end = 1.0, 0.10
        else:
           self.tau_start, self.tau_end = 0.1, None

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature_only(self.adata)

        if 'adj' not in adata.obsm.keys():
            if self.ST == 'Visium':
                print("Constructing adjacency matrix with", neighbors, "rings...")
                construct_connectivity_graph(self.adata, n_rings=neighbors)      
            elif self.ST == 'HD':
                print("Constructing adjacency matrix with", neighbors, "rings...")
                construct_connectivity_graph(self.adata, n_rings=neighbors, n_neighs=4) 
                # print(self.adata.obsm['adj'].sum(axis=1).mean())
                # print(np.median(self.adata.obsm['adj'].sum(axis=1)))
            else:
                print("Constructing adjacency matrix with", neighbors, "neighbors...")
                construct_interaction_KNN(self.adata, n_neighbors=neighbors)
                # print("Adjacency matrix is constructed with", neighbors, "neighbors.")
                # print(self.adata.obsm['adj'].sum(axis=1).mean())

        
        self._prepare_data(dim_output, metric=metric)   
        n_nodes = self.adata.shape[0]


    
    def train(self):
    
        if 'learn' in self.data:
            return self._train_learn_t()
        else:
            return self._train()

    def _train(self):

        n_nodes = self.adata.shape[0]
        self.model = model_GAT_2L_multi_learn_t(self.dim_input, self.dim_output, gnn_layer=self.gnn_layer, threshold=self.t, data=self.data, n_nodes=n_nodes).to(self.device)
        self.model = self.model.to(self.dtype)
        self.loss_fn = self._get_loss_function(self.data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                            weight_decay=self.weight_decay)


        self.edge_scores = torch.nonzero(torch.Tensor(self.edge_scores).to(self.device), as_tuple=False).t()
        print(f"Begin to train with {self.data} data!")

        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
                
            self.output = self.model(self.x, self.edge_scores, self.edge_scores)
            loss = self.loss_fn()
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            if self.verbose:
                print(f"Epoch {epoch+1} | loss = {loss.item():.8f}")

        print("Optimization finished for ST data!")
        with torch.no_grad():
            self.model.eval()


            self.output = self.model(self.x, self.edge_scores, self.edge_scores)
            self.z = self.get_latent().detach().cpu().numpy()
            self.adata.obsm['emb'] = self.z

        del self.z
        del self.output
        del self.x
        
        return self.adata

    def _train_learn_t(self):

        self._get_row_max_min_init_t(self.edge_scores, n_init=self.init_t)
        # degree = (self.edge_scores >= self.t.numpy()[:,np.newaxis]).sum(axis=1)
        # ratio = (degree >= 2).sum().item() / degree.shape[0]

        self.edge_scores = torch.FloatTensor(self.edge_scores).to(self.device).to(self.dtype)
        n_nodes = self.adata.shape[0]  
        self.model = model_GAT_2L_multi_learn_t(self.dim_input, self.dim_output, gnn_layer=self.gnn_layer, threshold=self.t, data=self.data, n_nodes=n_nodes).to(self.device)
        self.model = self.model.to(self.dtype)
        self.loss_fn = self._get_loss_function(self.data)   
        main_params = [p for n, p in self.model.named_parameters() if n != "encoder.t"]
        self.optimizer = torch.optim.Adam(main_params, lr=self.learning_rate)
        self.optimizer_t = torch.optim.Adam([self.model.encoder.t], lr=self.learning_rate_t)
        self.scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_t, T_max=self.epochs)

        self.model.tau = self.tau_start
        PATH = '/home/mongardi/spatial_biology/SpaTIM/src/ckpt'

        max_val = torch.tensor(self.max_spot_dist, device=self.device, dtype=self.dtype)
        min_val = torch.tensor(self.min_spot_dist, device=self.device, dtype=self.dtype)
       

        self.n_nodes = []
        counter = 0
        self.model.encoder.t.requires_grad_(True)
        for epoch in tqdm(range(self.epochs)): 

            self.t_prev = self.model.encoder.t.clone()
            self.d_prev = (self.edge_scores + 1e-6 >= self.t_prev.view(-1, 1)).sum(dim=1).float()
            self.model.train()
            self.model.training_t = True

            self.output = self.model(self.x, None, self.edge_scores, None) #, self.edge_scores 
            loss = self.loss_fn()
            self.optimizer.zero_grad()
            self.optimizer_t.zero_grad()
            if self.verbose:
                print(f"Epoch {epoch+1} | loss = {loss.item():.8f}")           
            loss.backward() 
            self.optimizer.step()
            self.optimizer_t.step()
            if self.schedule_t:
                self.scheduler_t.step()
            
            with torch.no_grad():
                t = self.model.encoder.t                   
                t.data = torch.max(torch.min(t, max_val), min_val)#torch.min(t, max_val)
                degree = (self.edge_scores + 1e-6 >= t.view(-1,1)).sum(dim=1).float()
                
                # if (epoch + 1) % 100 == 0 or epoch==0:
                #     plt.figure()
                #     plt.hist(degree.cpu().numpy(), bins=50)
                #     os.makedirs(f"/home/mongardi/spatial_biology/SpaTIM/plots/{self.slice}", exist_ok=True)
                #     plt.savefig(f"/home/mongardi/spatial_biology/SpaTIM/plots/{self.slice}/degree_epoch_"+str(epoch + 1)+'.png')
                            
                self.n_nodes.append(degree.sum().item())
                delta_t = torch.norm(t.data - self.t_prev) / torch.norm(self.t_prev)
                delta_degree = torch.norm(degree - self.d_prev) / torch.norm(self.d_prev)
                #print(degree.sum().item(), delta_t.item(), delta_degree.item())
                if delta_degree.item() < 5e-3 and epoch > 200:
                    counter = counter + 1
                    
                    if counter >= 50:
                        self.model.load_state_dict(prev_state)
                        break
                else:
                    counter = 0
                    prev_state = deepcopy(self.model.state_dict())
    
            self.model.tau = self._tau_schedule_new(epoch, type=self.annealing)
      
        print("Fine-tuning the model for learned threshold...")

        self.model.encoder.t.requires_grad_(False)
        self.final_adj =  (self.edge_scores.to_dense() + 1e-6 >= self.model.encoder.t.view(-1,1)).int()
        print('slice: ', self.slice, self.final_adj.sum())
        self.final_adj = self.final_adj.to_sparse()
        self.final_adj_index = self.final_adj.indices().to(self.device)
        self.final_edge_scores= self.final_adj.values().to(self.device)

        self.model.training_t = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), 1e-4, 
                                             weight_decay=self.weight_decay)

        counter = 0                                     
        for epoch in tqdm(range(self.ft_epochs)):

            self.prev_state = deepcopy(self.model.state_dict())
            self.model.train()
            self.output = self.model(self.x, self.final_adj_index, self.final_edge_scores)
           
            if self.graph_loss:
                self.loss_feat = self.loss_fn()
                self.loss_adj = self._compute_loss_adj()
                loss = self.loss_feat + 0.005* self.loss_adj
                
            else:
                
                loss = self.loss_fn()

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            delta_weights = compute_grad_norm(self.model)
            #print(f"Epoch {epoch+1} | loss = {loss.item():.8f} | feat_loss = {self.loss_feat.item():.8f} ", delta_weights)
            #print(delta_weights)
            if delta_weights < 1e-4:
                counter = counter + 1
                
                if counter >= 50:
                    self.model.load_state_dict(prev_state)
                    break
            else:
                counter = 0
                prev_state = deepcopy(self.model.state_dict())
            
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            self.output = self.model(self.x, self.final_adj_index, self.final_edge_scores)
            self.z = self.get_latent().detach().cpu().numpy()
            self.adata.obsm['emb'] = self.z 
      

        del self.z
        del self.output
        del self.x

        return self.adata

    def _load_features(self):
  
        feats = {}
        for key in ["feat", "flux", "met"]:
            if key in self.adata.obsm:
                feats[key] = torch.as_tensor(
                    self.adata.obsm[key].copy(),
                    dtype=self.dtype,
                    device=self.device
                )
        return feats


    def _prepare_data(self, dim_output, metric="cosine"):

        feats = self._load_features()
        print("Removing isolated spots...")
        degree = self.adata.obsm["adj"].sum(axis=1)
        mask = degree == 0
        if mask.sum() > 0:
            print("Removing isolated spots...")
            self.adata[~mask].copy()

        if self.data == "ST":
            self.features = feats["feat"]
            self.dim_input = self.features.shape[1]
            self.dim_output = dim_output
            self.x = self.features

        elif self.data == "ST_learn":
            self.features = feats["feat"]
            self.dim_input = self.features.shape[1]
            self.dim_output = dim_output
            self.x = self.features
        
            if "spot_image" in self.adata.obsm.keys():
                distance_matrix = compute_distance_matrix(self.adata.obsm["spot_image"].squeeze(1), metric=metric)
            # stochastic gene sampling already computed        
            elif "sgs" in self.adata.obsm.keys():
                distance_matrix = self.adata.obsm["sgs"]

            # compute stochastic gene sampling on the fly (roughly 1 minutes for 30,000 spots and all genes) 
            elif 'learn' in self.data:
               distance_matrix = self._compute_sgs_matrix(k_iter=self.k_iter, n_iter=self.n_iter, scaling=self.scaling, drop_housekeeping=self.drop_housekeeping)

            else:
                print("No distance matrix found for learning t!")
    
        self.edge_scores = preprocess_adj(self.adata.obsm["adj"], normalize=False)
        if 'learn' in self.data:
            if "spot_image" in  self.adata.obsm.keys():
                self.edge_scores = self.edge_scores * (1 - distance_matrix)
            else:
                self.edge_scores = self.edge_scores * distance_matrix

    def _tau_schedule(self, epoch, type='cosine'):
        
        if type == 'cosine':    
            frac = min(epoch / self.epochs, 1.0)
            return self.tau_end + 0.5*(self.tau_start - self.tau_end)*(1 + math.cos(math.pi*frac))
        elif type == 'linear':
            return self.tau_start * ((self.tau_end / self.tau_start) ** (epoch / self.epochs))
        else:
            return self.tau_start

    def _tau_schedule_new(self, epoch, type='cosine', max_epoch=100):
        
        if type == 'cosine':    
            if epoch <= max_epoch:
                frac = min(epoch / max_epoch, 1.0)
                return self.tau_end + 0.5*(self.tau_start - self.tau_end)*(1 + math.cos(math.pi*frac))
            else:
                return self.model.tau

        elif type == 'linear':
            if epoch <= max_epoch:
                return self.tau_start * ((self.tau_end / self.tau_start) ** (epoch / max_epoch))
            else:
                return self.model.tau
        else:
            return self.tau_start

    def _compute_loss_ST(self):
        _, x_recon = self.output
        return F.mse_loss(self.x, x_recon)

    def _compute_loss_adj(self):
        
        z, _ = self.output
        return self.model.compute_adj_loss(z, self.final_adj_index)

    def _compute_degree_loss(self, min_deg=4):
        
        degree = (self.edge_scores.to_dense() > self.model.encoder.t).sum(dim=1).float()
        loss = F.relu(min_deg - degree).pow(2).mean()
        return loss

    def _get_loss_function(self, data):
        loss_functions = {
            'ST': self._compute_loss_ST,
            'ST_learn': self._compute_loss_ST,
        }
        if self.adj_reconstruction:
            return loss_functions[data] + self._compute_loss_adj()
        else: 
            return loss_functions[data]

    def get_latent(self):
        if self.data == 'ST':
            h, _ = self.output
        elif self.data == 'ST_learn':
            h, _ = self.output
        return h

    def _get_row_max_min_init_t(self, matrix, n_init=6):

        m_masked = matrix.copy()
        np.fill_diagonal(m_masked, np.nan)     
        m_masked[m_masked == 0.0] = np.nan
        self.max_spot_dist = np.nanmax(m_masked, axis=1)
        self.max_spot_dist = np.nan_to_num(self.max_spot_dist, nan=0.0)
        self.min_spot_dist = np.nanmin(m_masked, axis=1)
        self.min_spot_dist = np.nan_to_num(self.min_spot_dist, nan=0.0)
        if self.min_spot_dist.min() == 0.0:
            self.min_spot_dist[self.min_spot_dist == 0.0] =  1e-2
        m_masked[np.isnan(m_masked)] = 0.0
        self.t = np.sort(m_masked, axis=1)[:, -(n_init)]
        self.t[self.t == 0.0] = self.max_spot_dist[self.t == 0.0]

        self.t = torch.tensor(self.t)
        # print(self.max_spot_dist, self.min_spot_dist)
        # print(self.t)

    def _compute_sgs_matrix(self, k_iter=30, n_iter=100, scaling=True, drop_housekeeping=True):

        hkg_dict = json.load(open(".data/HOUNKPE_HOUSEKEEPING_GENES.v2025.1.Hs.json"))
        hkg = hkg_dict['HOUNKPE_HOUSEKEEPING_GENES']['geneSymbols']
        hkg_ids = [self.adata.var_names.get_loc(gene) for gene in hkg if gene in self.adata.var_names]
        distance_matrix = stochastic_similarity_matrix_torch_adj_v2(
                        self.adata.X.toarray(),
                        self.adata.obsm["adj"], 
                        k=k,
                        n_iter=n_iter,
                        gene_weights_init=gene_weights,
                        random_state=42, 
                        scaling=scaling, 
                        drop_housekeeping=drop_housekeeping,
                        house_keeping_genes = hkg_ids, 
                        device='cpu'
                    )
        return distance_matrix