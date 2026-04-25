import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
import torch.nn.functional as F
from torch_geometric.nn import GCNConv as GraphConv
from .myGAT import GATv2Conv as GATConv
from torch_geometric import seed_everything
from torch_geometric.utils import negative_sampling, dense_to_sparse, add_self_loops, to_dense_adj
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
seed_everything(12345)

EPS = 1e-15

### GAT layer
class model_GAT(torch.nn.Module):

    def __init__(self, hidden_dims, graph=False, bias=False):
        super(model_GAT, self).__init__()

        self.graph = graph                 
        [in_dim, num_hidden, out_dim] = hidden_dims
        num_hidden, out_dim =  64, 32 #128, 64
        dr = 0.0
        if self.graph:
          
            self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)                     
            self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)
            self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                                dropout=0, add_self_loops=False, bias=False)

        else:

            self.conv1 = nn.Linear(in_dim, num_hidden*2, bias=bias)
            self.conv2 = GATConv(num_hidden*2, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv2_2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv3_2 = GATConv(num_hidden, num_hidden*2, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv4 = nn.Linear(num_hidden*2, in_dim, bias=bias)        

    def forward(self, features, edge_index):
        if self.graph:

            h1 = F.elu(self.conv1(features, edge_index))
            h2 = self.conv2(h1, edge_index)
            h3 = F.elu(self.conv3(h2, edge_index))
            h4 = self.conv4(h3, edge_index)
        
        else:
            h1 = F.elu(self.conv1(features))
            h2 = F.elu(self.conv2(h1, edge_index))
            h2 = self.conv2_2(h2, edge_index)
            h3 = F.elu(self.conv3(h2, edge_index))
            h3 = F.elu(self.conv3_2(h3, edge_index))
            h4 = self.conv4(h3)

        return h2, h4

class model_GAT_t_node(torch.nn.Module):
    def __init__(self, hidden_dims, threshold, n_nodes, bias= False, graph=True, multi_scale=False):
        super(model_GAT_t_node, self).__init__()
        
        self.graph = graph
        self.multi_scale = multi_scale
        dr = 0.0
        [in_dim, num_hidden, out_dim] = hidden_dims
        out_dim = 32 #64
        num_hidden = 64 #128
        #print(num_hidden, out_dim)

        if self.graph:
            self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=dr, add_self_loops=False, bias=bias)
            self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=dr, add_self_loops=False,  bias=bias)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
        else:
  
            self.conv1 = nn.Linear(in_dim, num_hidden*2, bias=bias)
            self.conv2 = GATConv(num_hidden*2, num_hidden, heads=1, concat=False,
                             dropout=dr, add_self_loops=False, bias=bias)
            self.conv2_2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=dr, add_self_loops=False, bias=bias)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv3_2 = GATConv(num_hidden, num_hidden*2, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv4 = nn.Linear(num_hidden*2, in_dim, bias=bias)


        self.t = nn.Parameter(threshold, requires_grad=True)
        self.eps = 1e-6
        self.dropout = nn.Dropout(p=0.0)
       
    def forward(self, features, edge_index, edge_scores, batch_id, tau, train=True):  

        
        # edge_score not sparse
        features = self.dropout(features)
        if train: 
            #print('GAT')
            edge_scores_binary = (edge_scores != 0).float()
            values_mask = torch.sigmoid((edge_scores - self.t.view(-1, 1)) / tau)
            values_mask = (values_mask * edge_scores_binary).to_sparse()
            edge_index_thr = values_mask.indices() 
            edge_attr = values_mask.values().unsqueeze(-1)

            # values_mask = torch.sigmoid((edge_scores - self.t[:, None]) / tau)
            # mask = edge_scores != 0
            # edge_index_thr = mask.nonzero(as_tuple=False).T
            # edge_attr = values_mask[mask].unsqueeze(-1)
            # values_mask = torch.sigmoid((edge_scores - self.t[:, None]) / tau)
            # values_mask = values_mask.masked_fill(edge_scores == 0, 0.0).to_sparse()
            # edge_index_thr = values_mask.indices()
            # edge_attr = values_mask.values().unsqueeze(-1)

            if self.graph:
                h1 = F.elu(self.conv1(features, edge_index_thr, edge_weight=edge_attr))
                h2 = self.conv2(h1, edge_index_thr, edge_weight=edge_attr)
                h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
                h4 = self.conv4(h3, edge_index_thr, edge_weight=edge_attr)
            else:
                # h1 = F.elu(self.conv1(features, edge_index_thr, edge_weight=edge_attr))
                # h2 = self.conv2(h1)
                # h3 = F.elu(self.conv3(h2))
                # h4 = self.conv4(h3, edge_index_thr, edge_weight=edge_attr)
               	h1 = F.elu(self.conv1(features))
                h2 = F.elu(self.conv2(h1, edge_index_thr, edge_weight=edge_attr))
                h2 = self.conv2_2(h2, edge_index_thr, edge_weight=edge_attr)
                h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
                h3 = F.elu(self.conv3_2(h3, edge_index_thr, edge_weight=edge_attr))
                h4 = self.conv4(h3)

  
        else:
         
        
            if self.graph:
                h1 = F.elu(self.conv1(features, edge_index, edge_weight=edge_scores))
                h2 = self.conv2(h1, edge_index, edge_weight=edge_scores)
                h3 = F.elu(self.conv3(h2, edge_index, edge_weight=edge_scores))
                h4 = self.conv4(h3, edge_index, edge_weight= edge_scores)
            else:
                # h1 = F.elu(self.conv1(features, edge_index, edge_weight=edge_scores))
                # h2 = self.conv2(h1)
                # h3 = F.elu(self.conv3(h2))
                # h4 = self.conv4(h3, edge_index, edge_weight= edge_scores)
             
                h1 = F.elu(self.conv1(features))
                h2 = F.elu(self.conv2(h1, edge_index, edge_weight=edge_scores))
                h2 = self.conv2_2(h2, edge_index, edge_weight=edge_scores)
                h3 = F.elu(self.conv3(h2, edge_index, edge_weight=edge_scores))
                h3 = F.elu(self.conv3_2(h3, edge_index, edge_weight=edge_scores))
                h4 = self.conv4(h3)
        
      
        return h2, h4


# class model_GAT_t_node(torch.nn.Module):
#     def __init__(self, hidden_dims, threshold, n_nodes):
#         super(model_GAT_t_node, self).__init__()
        
#         [in_dim, num_hidden, out_dim] = hidden_dims
#         print(in_dim, num_hidden, out_dim)
#         self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
#         self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
#                              dropout=0, add_self_loops=False, bias=False)
    
#         self.t = nn.Parameter(threshold, requires_grad=True)
#         self.eps = 1e-6
 
          
#     def forward(self, features, edge_index, edge_scores, batch_id, tau, train=True):
        
        
        
#         if train: 

#             edge_scores_binary = (edge_scores != 0).float()
#             #values_mask = torch.sigmoid((edge_scores - self.t.unsqueeze(1)) / tau)
#             values_mask = torch.sigmoid((edge_scores - self.t.view(-1, 1)) / tau)
#             values_mask = (values_mask * edge_scores_binary).to_sparse()
#             #values_mask = torch.maximum(values_mask, values_mask.T)
#             edge_index_thr = values_mask.indices() 
#             values_mask = values_mask.values()
#             edge_attr = values_mask.unsqueeze(-1)

#             h1 = F.elu(self.conv1(features, edge_index_thr, edge_weight=edge_attr))
#             h2 = self.conv2(h1, edge_index_thr, edge_weight=edge_attr)
#             h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
#             h4 = self.conv4(h3, edge_index_thr, edge_weight=edge_attr)
    

#         else:
         
#             h1 = F.elu(self.conv1(features, edge_index, edge_weight=edge_scores))
#             h2 = self.conv2(h1, edge_index, edge_weight=edge_scores)
#             h3 = F.elu(self.conv3(h2, edge_index, edge_weight=edge_scores))
#             h4 = self.conv4(h3, edge_index, edge_weight= edge_scores)
        

#         return h2, h4

class model_GAT_t_node_batch(torch.nn.Module):
    def __init__(self, hidden_dims, threshold, n_nodes, graph=False, bias=False):
        super(model_GAT_t_node_batch, self).__init__()
        
        dr = 0.0
        [in_dim, num_hidden, out_dim] = hidden_dims
        print(in_dim, num_hidden, out_dim)
        self.graph = graph 
        if self.graph:
            self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=dr, add_self_loops=False, bias=bias)
            self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=dr, add_self_loops=False,  bias=bias)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
        else:

            self.conv1 = nn.Linear(in_dim, num_hidden*2, bias=bias)
            self.conv2 = GATConv(num_hidden*2, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv2_2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv3_2 = GATConv(num_hidden, num_hidden*2, heads=1, concat=False,
                                dropout=dr, add_self_loops=False, bias=bias)
            self.conv4 = nn.Linear(num_hidden*2, in_dim, bias=bias)
    
        self.eps = 1e-6
        self.t = nn.Embedding(n_nodes, 1)
        
        with torch.no_grad():
            
                self.t.weight.copy_(threshold.unsqueeze(1))
          
    def forward(self, features, edge_index, edge_scores, batch_nids, tau, train=True):
        
        if train: 
            
     
            t_batch = self.t(batch_nids)
            #edge_scores_binary = (edge_scores != 0).float()
            values_mask = torch.sigmoid((edge_scores - t_batch.view(-1, 1)) / tau)
            #values_mask = (values_mask * edge_scores_binary).to_sparse()

            # edge_index_thr = values_mask.indices() 
            # values_mask = values_mask.values()
            # edge_attr = values_mask.unsqueeze(-1)

            mask = edge_scores != 0
            edge_index_thr = mask.nonzero(as_tuple=False).T
            edge_attr = values_mask[mask].unsqueeze(-1)
           

            if self.graph:
                h1 = F.elu(self.conv1(features, edge_index_thr, edge_weight=edge_attr))
                h2 = self.conv2(h1, edge_index_thr, edge_weight=edge_attr)
                h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
                h4 = self.conv4(h3, edge_index_thr, edge_weight=edge_attr)
            else:
                h1 = F.elu(self.conv1(features))
                h2 = F.elu(self.conv2(h1, edge_index_thr, edge_weight=edge_attr))
                h2 = self.conv2_2(h2, edge_index_thr, edge_weight=edge_attr)
                h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
                h3 = F.elu(self.conv3_2(h3, edge_index_thr, edge_weight=edge_attr))
                h4 = self.conv4(h3)
        
        else:   
       
            if self.graph:
                h1 = F.elu(self.conv1(features, edge_index))
                h2 = self.conv2(h1, edge_index)
                h3 = F.elu(self.conv3(h2, edge_index))
                h4 = self.conv4(h3, edge_index)
            else:
                h1 = F.elu(self.conv1(features))
                h2 = F.elu(self.conv2(h1, edge_index))
                h2 = self.conv2_2(h2, edge_index)
                h3 = F.elu(self.conv3(h2, edge_index))
                h3 = F.elu(self.conv3_2(h3, edge_index))
                h4 = self.conv4(h3)
        
        return h2, h4


### GCN layer
### to do: add batch and graph options similar to GAT
class model_GCN(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(model_GCN, self).__init__()

        
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GraphConv(in_dim, num_hidden)
        self.conv2 = GraphConv(num_hidden, out_dim)
        self.conv3 = GraphConv(out_dim, num_hidden)
        self.conv4 = GraphConv(num_hidden, in_dim)
        print('GCN forward')

    def forward(self, features, edge_index):

        h1 = F.elu(self.conv1(features, edge_index))
        h2 = self.conv2(h1, edge_index)
        h3 = F.elu(self.conv3(h2, edge_index))
        h4 = self.conv4(h3, edge_index)

        return h2, h4

class model_GCN_t_node(torch.nn.Module):
    def __init__(self, hidden_dims, threshold, n_nodes):
        super(model_GCN_t_node, self).__init__()
        
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.conv1 = GraphConv(in_dim, num_hidden)
        self.conv2 = GraphConv(num_hidden, out_dim)
        self.conv3 = GraphConv(out_dim, num_hidden)
        self.conv4 = GraphConv(num_hidden, in_dim)
    
        self.t = nn.Parameter(threshold, requires_grad=True)
        self.eps = 1e-6
        #print('Initial threshold:', self.t.item())
          
    def forward(self, features, edge_index, edge_scores, batch_id, tau, train=True):

        if train: 

            edge_scores_binary = (edge_scores != 0).float()
            #values_mask = torch.sigmoid((edge_scores - self.t.unsqueeze(1)) / tau)
            values_mask = torch.sigmoid((edge_scores - self.t.view(-1, 1)) / tau)
            #print('values_mask', values_mask)
            values_mask = (values_mask * edge_scores_binary).to_sparse()
            #values_mask = torch.maximum(values_mask, values_mask.T)
            edge_index_thr = values_mask.indices() 
            values_mask = values_mask.values()
            #print(values_mask.shape)
            edge_attr = values_mask.unsqueeze(-1)

            h1 = F.elu(self.conv1(features, edge_index_thr, edge_weight=edge_attr))
            h2 = self.conv2(h1, edge_index_thr, edge_weight=edge_attr)
            #h2 = F.elu(h2)
            h3 = F.elu(self.conv3(h2, edge_index_thr, edge_weight=edge_attr))
            h4 = self.conv4(h3, edge_index_thr, edge_weight=edge_attr)

        else:
           
            h1 = F.elu(self.conv1(features, edge_index))
            h2 = self.conv2(h1, edge_index)
            #h2 = F.elu(h2)
            h3 = F.elu(self.conv3(h2, edge_index))
            h4 = self.conv4(h3, edge_index)
             
        return h2, h4



class model_GAT_2L_multi_learn_t(nn.Module):

    def __init__(self, in_channels, out_channels, gnn_layer='GCN', threshold=None, data='all', tau=0.1, training_t=False, n_nodes=None, batch=False):
        super(model_GAT_2L_multi_learn_t, self).__init__()

    
        self.tau = tau
        self.training_t = training_t
        if data=='all':
            # implenting integration of other modalities, future work 
            raise NotImplementedError
            
        elif data=='ST':
            if gnn_layer=='GAT':
                self.encoder = model_GAT([in_channels, out_channels*2, out_channels])
            elif gnn_layer=='GCN':
                self.encoder = model_GCN([in_channels, out_channels*2, out_channels])
            else:
                raise ValueError('Unsupported GNN layer type')

        elif data=='ST_learn':
            
            if gnn_layer=='GAT':
                if batch:
                    print('learning t with batch')
                    self.encoder = model_GAT_t_node_batch([in_channels, out_channels*2, out_channels], threshold, n_nodes=n_nodes)
                else:
                    self.encoder = model_GAT_t_node([in_channels, out_channels*2, out_channels], threshold, n_nodes=n_nodes)
            elif gnn_layer=='GCN':
                if batch:
                    self.encoder = model_GCN_t_node_batch([in_channels, out_channels*2, out_channels], threshold, n_nodes=n_nodes)
                else:
                    self.encoder = model_GCN_t_node([in_channels, out_channels*2, out_channels], threshold, n_nodes=n_nodes)
            else:
                raise ValueError('Unsupported GNN layer type')

        print('Building model...')
        self.forward_function = self._get_forward_function(data)

    def _forward_ST(self, x, adj, edge_scores, batch_ids):
        hidden_feat, emb = self.encoder(x, adj)
        return hidden_feat, emb

    def _forward_ST_learn(self, x, adj, edge_scores, batch_ids):
        hidden_feat, emb = self.encoder(x, adj, edge_scores, batch_ids, self.tau, self.training_t)
        return hidden_feat, emb
    
    
    def _get_forward_function(self, data):
        forward_functions = {
            'ST': self._forward_ST,
            'ST_learn': self._forward_ST_learn,

        }
        return forward_functions[data]

    def _reconstruct_adjacency(self, z, adj, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

    def compute_adj_loss(self, z,  pos_edge_index, neg_edge_index=None):
        sum_loss = 0
        if isinstance(z, list):
            for emb in z:
                pos_loss = -torch.log(
                self._reconstruct_adjacency(z, pos_edge_index, sigmoid=True) + EPS).mean()

                if neg_edge_index is None:
                    neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
                neg_loss = -torch.log(1 -
                                    self._reconstruct_adjacency(z, neg_edge_index, sigmoid=True) +
                                    EPS).mean()
            sum_loss += pos_loss + neg_loss

        else:
            pos_loss = -torch.log(
                self._reconstruct_adjacency(z, pos_edge_index, sigmoid=True) + EPS).mean()

            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
            neg_loss = -torch.log(1 -
                                  self._reconstruct_adjacency(z, neg_edge_index, sigmoid=True) +
                                  EPS).mean()
            sum_loss += pos_loss + neg_loss
            
        return sum_loss

    def set_other_params(self,):
        self.alpha = alpha
        self.t = t

    def forward(self, x, adj, edge_scores=None, batch_ids=None) -> torch.Tensor:
        x = self.forward_function(x, adj, edge_scores, batch_ids)
        return x

