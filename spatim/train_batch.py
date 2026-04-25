import torch
import math
from preprocess import * #preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, add_contrastive_label, get_feature, permutation, fix_seed, construct_interaction_corr, combine_graphs, construct_interaction_radius, construct_interaction_refine_image, construct_interaction_refine_image_subspot, construct_connectivity_graph, construct_connectivity_graph_refine_image
import time
import sys
import random
import numpy as np
from model_utils import Encoder, EarlyStopper_loss, EarlyStopper_weights, EarlyStopper_grad, compute_grad_norm
from model import *
from utils_b import plot_UMAP
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse import csc_matrix
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric import seed_everything
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import json
from utils_b import stochastic_similarity_matrix_torch_adj_v2

class Model():

    def __init__(self, 
        adata,
        slice, 
        data = "ST",
        gnn_layer = 'GCN',
        device= torch.device('cpu'),
        learning_rate=0.001,
        learning_rate_t=5e-4,
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
        graph_loss = True, 
        init_t = 6,  
        schedule_t = False, 
        dtype= torch.float64,
        batch_size = 4096, 
        accumulate = 1, 
        ST='Visium'
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
        self.batch_size = batch_size
        self.accumulate = accumulate
        self.ST = ST

        if self.annealing in ['cosine', 'linear']:
            self.tau_start, self.tau_end = 1.0, 0.10
        else:
           self.tau_start, self.tau_end = 0.1, None

        fix_seed(self.random_seed)

        if 'highly_variable' not in adata.var.keys():
            preprocess(self.adata)

        if 'feat' not in adata.obsm.keys():
            get_feature_only(self.adata, normalize=True)
            print("Features are extracted for N spots:", self.adata.shape[0])
            print("Number of train batches:", math.ceil(self.adata.shape[0]/self.batch_size))

        if 'adj' not in adata.obsm.keys():
            print("Constructing adjacency matrix with", neighbors, "neighbors...")
            if self.ST == 'Visium':
                construct_connectivity_graph(self.adata, n_rings=neighbors)      
            elif self.ST == 'HD':
                construct_connectivity_graph(self.adata, n_rings=neighbors, n_neighs=4) 
                print(self.adata.obsm['adj'].sum(axis=1).mean())
                print(np.median(self.adata.obsm['adj'].sum(axis=1)))
            else:
                construct_interaction(self.adata, n_neighbors=neighbors)
                print("Adjacency matrix is constructed with", neighbors, "neighbors.")
                print(self.adata.obsm['adj'].sum(axis=1).mean())
        
        self._prepare_data(dim_output, metric=metric)   
        n_nodes = self.adata.shape[0]
            
    
    def train(self):
    
        if 'learn' in self.data:
            return self._train_learn_t()
        else:
            return self._train()

    def _train(self):

        loader, test_loader = self._create_batch()
        n_nodes = self.adata.shape[0]
        self.model = model_GAT_2L_multi_learn_t(self.dim_input, self.dim_output, gnn_layer=self.gnn_layer, threshold=self.t, data=self.data, n_nodes=n_nodes).to(self.device)
        self.model = self.model.to(self.dtype)
        self.loss_fn = self._get_loss_function(self.data)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                            weight_decay=self.weight_decay)
        
        self.accumulate = len(loader)
        print(f"Begin to train with {self.data} data!")
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()

            self.optimizer.zero_grad()

            for batch_id, mini_batch in enumerate(loader):
                x_batch = mini_batch.x.to(self.device).to(self.dtype)          
                edge_index_batch = mini_batch.edge_index.to(self.device)
                n_ids = mini_batch.n_id                 

                self.output = self.model(x_batch, edge_index_batch)
                self.loss_feat = F.mse_loss(
                    self.output[1][:mini_batch.batch_size],
                     x_batch[:mini_batch.batch_size])
                del self.output
                loss = self.loss_feat /self.accumulate
                print('Feature loss: ', self.loss_feat.item())
                loss.backward() 
                if (batch_id + 1) % self.accumulate == 0 or (batch_id + 1) == len(loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
              
        with torch.no_grad():
            self.model.eval()
            for batch_id, mini_batch in enumerate(test_loader):
                x_batch = mini_batch.x.to(self.device).to(self.dtype)          
                edge_index_batch = mini_batch.edge_index.to(self.device)
                n_ids = mini_batch.n_id                            
                c_n_ids = n_ids[:mini_batch.batch_size]
                output_batch = self.model(x_batch, edge_index_batch)
                if batch_id == 0:
                    z_batch = output_batch[0][:mini_batch.batch_size].detach().cpu()
                    x_recon_batch = output_batch[1][:mini_batch.batch_size].detach().cpu()
                else:
                    z_batch = torch.cat((z_batch, output_batch[0][:mini_batch.batch_size].detach().cpu()), dim=0)
                    x_recon_batch = torch.cat((x_recon_batch, output_batch[1][:mini_batch.batch_size].detach().cpu()), dim=0)

            self.adata.obsm['emb'] = z_batch.numpy()
            self.adata.obsm['x_recon'] = x_recon_batch.numpy()

        del self.x
        return self.adata

    def _train_learn_t(self):

        loader, test_loader = self._create_batch()
        self._get_row_max_min_init_t(self.edge_scores, n_init=self.init_t)
        self.edge_scores = torch.FloatTensor(self.edge_scores).to(self.dtype).to_sparse().to(self.device)#.to(self.device).to(self.dtype)
        n_nodes = self.adata.shape[0]  
        self.model = model_GAT_2L_multi_learn_t(self.dim_input, self.dim_output, gnn_layer=self.gnn_layer, threshold=self.t, batch=True, data=self.data, n_nodes=n_nodes).to(self.device)
        self.model = self.model.to(self.dtype)
        self.loss_fn = self._get_loss_function(self.data)   
        main_params = [p for n, p in self.model.named_parameters() if n != "encoder.t.weight"]
        self.optimizer = torch.optim.Adam(main_params, lr=self.learning_rate)
        self.optimizer_t = torch.optim.Adam(self.model.encoder.t.parameters(), lr=self.learning_rate_t)
        self.scheduler_t = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_t, T_max=self.epochs)

        self.model.tau = self.tau_start
        print(self.tau_start, self.model.tau)
        PATH = '/home/mongardi/spatial_biology/SpaTIM/src/ckpt'
        
        max_val = torch.tensor(self.max_spot_dist, device=self.device, dtype=self.dtype).view(-1, 1)
        min_val = torch.tensor(self.min_spot_dist, device=self.device, dtype=self.dtype).view(-1, 1)
        print(min_val[min_val==0.0])
        #sys.exit(0)
        self.n_nodes = []
        counter = 0
        self.model.encoder.t.requires_grad_(True)
        self.accumulate = len(loader)
        for epoch in tqdm(range(self.epochs)): 

            self.model.train()
            self.model.training_t = True
            self.optimizer.zero_grad()
            self.optimizer_t.zero_grad()

            for batch_id, mini_batch in enumerate(loader):
            
                x_batch = mini_batch.x.to(self.device).to(self.dtype)          
                edge_index_batch = mini_batch.edge_index.to(self.device)
                n_ids = mini_batch.n_id .to(self.device)                           
                c_n_ids = n_ids[:mini_batch.batch_size].to(self.device)
                batch_edge_scores = self._get_dense_batch_edge_scores(
                    self.edge_scores,
                    c_n_ids,
                    n_ids,
                    self.device
                )
                                
            
                output = self.model(x_batch, edge_index_batch, batch_edge_scores, c_n_ids)
                loss_feat = F.mse_loss(output[1][:mini_batch.batch_size], x_batch[:mini_batch.batch_size])
                print('Feature loss: ', loss_feat.item())
                loss = loss_feat /self.accumulate
                
                batch_edge_scores = batch_edge_scores.detach()
                del output
                del batch_edge_scores
                del x_batch
                del edge_index_batch

                loss.backward() 
                if (batch_id + 1) % self.accumulate == 0 or (batch_id + 1) == len(loader):
                    self.optimizer.step()
                    self.optimizer_t.step()
                    self.optimizer.zero_grad()
                    self.optimizer_t.zero_grad()

            if self.schedule_t:
                self.scheduler_t.step()
                
            with torch.no_grad():
                t = self.model.encoder.t.weight             
                t.clamp_(min=min_val, max=max_val)

                self.model.tau = self._tau_schedule_new(epoch, type=self.annealing)
      
        print("Fine-tuning the model for learned threshold...")
        del loader, test_loader
        self.model.encoder.t.requires_grad_(False)
        with torch.no_grad():
            edge_score_cpu = self.edge_scores.cpu().to_dense()
            del self.edge_scores
            self.final_adj =  (edge_score_cpu + 1e-6 >= self.model.encoder.t.weight.detach().cpu().view(-1,1)).int()
     
        print('slice: ', self.slice, self.final_adj.sum().item())
        degree = self.final_adj.sum(dim=1).float()
        plt.figure()
        plt.hist(degree, bins=50)
        plt.savefig(f"/home/mongardi/spatial_biology/SpaTIM/src_final/degree_epoch_"+str(epoch + 1)+f"_{self.tau_start}.png")
        plt.close()
        loader, test_loader = self._create_batch(fine_tune=True)
        self.model.training_t = False
        main_params = [p for n, p in self.model.named_parameters() if n != "encoder.t.weight"]
        self.optimizer = torch.optim.Adam(main_params, lr=1e-4)
        counter = 0       
        for epoch in tqdm(range(self.ft_epochs)):

            #self.prev_state = deepcopy(self.model.state_dict())
            self.model.train()
            self.optimizer.zero_grad()
            for batch_id, mini_batch in enumerate(loader):
                
                x_batch = mini_batch.x.to(self.device).to(self.dtype)          
                edge_index_batch = mini_batch.edge_index.to(self.device)
                n_ids = mini_batch.n_id                            
                c_n_ids = n_ids[:mini_batch.batch_size]
                #batch_edge_scores = torch.ones(edge_index_batch.shape[1], device=self.device)
                self.output = self.model(x_batch, edge_index_batch, None)
                self.loss_feat = F.mse_loss(self.output[1][:mini_batch.batch_size], x_batch[:mini_batch.batch_size])
                loss = self.loss_feat / self.accumulate
                
                del self.output
                loss.backward() 
                if (batch_id + 1) % self.accumulate == 0 or (batch_id + 1) == len(loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                   
                # else:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad()    
       
            # delta_weights = compute_grad_norm(self.model)
            # print(delta_weights)
            # if delta_weights < 1e-4:
            #     counter = counter + 1
            #     if counter >= 50:
            #         #self.model.load_state_dict(prev_state)
            #         break
            # else:
            #     counter = 0
            #     prev_state = deepcopy(self.model.state_dict())
            
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            for batch_id, mini_batch in enumerate(test_loader):

                x_batch = mini_batch.x.to(self.device).to(self.dtype)          
                edge_index_batch = mini_batch.edge_index.to(self.device)
                n_ids = mini_batch.n_id                            
                c_n_ids = n_ids[:mini_batch.batch_size]
                #batch_edge_scores = torch.ones(edge_index_batch.shape[1], device=self.device)
                output_batch = self.model(x_batch, edge_index_batch, None)
                if batch_id == 0:
                    z_batch = output_batch[0][:mini_batch.batch_size].detach().cpu()
                    x_recon_batch = output_batch[1][:mini_batch.batch_size].detach().cpu()
                else:
                    z_batch = torch.cat((z_batch, output_batch[0][:mini_batch.batch_size].detach().cpu()), dim=0)
                    x_recon_batch = torch.cat((x_recon_batch, output_batch[1][:mini_batch.batch_size].detach().cpu()), dim=0)


            self.adata.obsm['emb'] = z_batch.numpy()
            self.adata.obsm['x_recon'] = x_recon_batch.numpy()
            print(z_batch.shape)

        del self.x
        del output_batch
        del self.model 
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
                distance_matrix = compute_distance_matrix_adj(self.adata.obsm["spot_image"].squeeze(1),
                    self.adata.obsm["adj"], metric=metric)

            elif "sgs" in self.adata.obsm.keys():
                distance_matrix = self.adata.obsm["sgs"]
          
            elif 'learn' in self.data:
               distance_matrix = self._compute_sgs_matrix()

            else:
    
                print("No distance matrix found for learning t!")
            #distance_matrix = compute_distance_matrix(self.adata.obsm['spot_image'].squeeze(1), metric=metric)
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

    def _create_batch(self, fine_tune=False):

        num_neighbors = [-1] 

        if fine_tune:
            
            if torch.is_tensor(self.final_adj):
                adj = self.final_adj
            else:
                adj = torch.from_numpy(self.final_adj)
            edge_index, edge_attr = dense_to_sparse(adj)
            data = Data(
                x=self.x,          
                edge_index=edge_index,          
                edge_attr=edge_attr)  
        else: 
            adj = torch.from_numpy(preprocess_adj(self.adata.obsm["adj"], normalize=False))
            edge_index, edge_attr = dense_to_sparse(adj)
            data = Data(
                x=self.x,          
                edge_index=edge_index,          
                edge_attr=edge_attr)  

        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=None,   
            shuffle=False)

        test_loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=None,
            shuffle=False)
        
        return loader, test_loader

    def _get_row_max_min_init_t(self, matrix, n_init=6):

        m_masked = matrix.copy()
        np.fill_diagonal(m_masked, np.nan)     
        self.max_spot_dist = np.nanmax(m_masked, axis=1)
        self.max_spot_dist = np.nan_to_num(self.max_spot_dist, nan=0.0)
        self.min_spot_dist = np.nanmin(m_masked, axis=1)
        self.min_spot_dist = np.nan_to_num(self.min_spot_dist, nan=0.0)
        if self.min_spot_dist.min() == 0.0:
            self.min_spot_dist[self.min_spot_dist == 0.0] =  1e-2
        self.t = np.sort(m_masked, axis=1)[:, -(n_init+1)]
        self.t[self.t == 0.0] = self.max_spot_dist[self.t == 0.0]

        self.t = torch.tensor(self.t)

    def _get_dense_batch_edge_scores(self, edge_scores, row_ids, col_ids, device):
        sub = edge_scores.index_select(0, row_ids)\
                     .index_select(1, col_ids)
        return sub.to_dense().to(device)

    
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