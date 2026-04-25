import os
import ot
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from scipy.spatial.distance import cdist, euclidean, cosine, mahalanobis
from scipy.optimize import linear_sum_assignment
import sklearn
from sklearn.preprocessing import MaxAbsScaler
import squidpy as sq
from squidpy._constants._pkg_constants import Key as sqKey


def construct_interaction_radius(adata, rad_cutoff=150):
    print(f"Building Graph with radius cutoff:{rad_cutoff}")
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    #nbrs = NearestNeighbors(n_neighbors=3+1).fit(position)  
    #_ , indices = nbrs.kneighbors(position)
    nbrs = NearestNeighbors(radius=rad_cutoff).fit(position)  
    distances, indices = nbrs.radius_neighbors(position, return_distance=True)
   
    interaction = np.zeros([n_spot, n_spot])
    # Fill adjacency matrix
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j: 
                interaction[i, j] = 1  
                
    adata.obsm['graph_neigh'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)

    print(np.sum(adj, axis=1).min(), np.sum(adj, axis=1).max(), np.sum(adj, axis=1).mean())
    adata.obsm['adj'] = adj


def construct_interaction(adata, n_neighbors=3):

    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')# W1 Wassertain distance, distance between two distrubutions, just euclidean?
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def mahalanobis_distance(vec1, vec2, inv_cov_matrix):
    diff = vec1 - vec2
    return np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))

def cosine_distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return 1 - dot_product / (norm_a * norm_b)

def compute_distance_matrix(matrix, metric='euclidean'):

    if metric == 'euclidean':
        distance_matrix = cdist(matrix, matrix, metric='euclidean')
    elif metric == 'mahalanobis':
        inv_cov_matrix = np.linalg.inv(np.cov(matrix.T))
        #distance_matrix = cdist(matrix, matrix, metric=lambda u, v: mahalanobis_distance(u, v, inv_cov_matrix)) # compute mahalanobis distance
        distance_matrix = cdist(matrix, matrix, metric='mahalanobis', VI=inv_cov_matrix) # compute mahalanobis distance
    elif metric == 'cosine':
        distance_matrix = cdist(matrix, matrix, metric='cosine')

    return distance_matrix

def compute_distance_matrix_adj(matrix, adj, metric='euclidean'):

    n_nodes = matrix.shape[0]
    distance_matrix = np.zeros((n_nodes, n_nodes))
    
    if metric == 'mahalanobis':
        inv_cov_matrix = np.linalg.inv(np.cov(matrix.T))

    rows, cols = np.where(adj == 1)
    for i, j in zip(rows, cols):
        if metric == 'euclidean':
            d = euclidean(matrix[i], matrix[j])
        elif metric == 'cosine':
            d = cosine(matrix[i], matrix[j])
        elif metric == 'mahalanobis':
            d = mahalanobis(matrix[i], matrix[j], inv_cov_matrix)

        distance_matrix[i, j] = d

    return distance_matrix


def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    
    adata.obsm['adj'] = adj
    print('Graph constructed!')   

def construct_connectivity_graph(adata, connectivity_key = None, n_rings=1, n_neighs=6):

    #print(f"Building Connectivity Graph with:{n_rings} rings")   
    connectivity_key = sqKey.obsp.spatial_conn(connectivity_key)
    sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=n_neighs)
    spatial_conn = adata.obsp[connectivity_key]
    spatial_conn = spatial_conn.toarray()
    adata.obsm['graph_neigh'] = spatial_conn
    adj = spatial_conn
    adj = adj + adj.T
    np.clip(adj, 0, 1, out=adj)
    #adj = np.where(adj>1, 1, adj)

    adata.obsm['adj'] = adj
    print('Graph constructed!')

def get_spots_spatial_connectivity(adata, connectivity_key = None, n_rings=1):

    print(f"Building Connectivity Graph with:{n_rings} rings")   
    connectivity_key = sqKey.obsp.spatial_conn(connectivity_key)
    sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=6)
    spatial_conn = adata.obsp[connectivity_key]
    spatial_conn = spatial_conn.toarray()
    return spatial_conn  

def load_flux_data(sample_id, preprocess=False, normalize=True):
    DATA_DIR = "/home/mongardi/spatial_biology/scFEA_data"
    if preprocess:
        file_name = sample_id + "_flux_prep.csv"
    else:
        file_name = sample_id + "_flux.csv"

    flux_data = pd.read_csv(os.path.join(DATA_DIR, file_name), index_col=0)

    if normalize:   
        new_flux_data = MaxAbsScaler().fit_transform(flux_data)
        df_flux_norm = pd.DataFrame(new_flux_data, columns=flux_data.columns, index=flux_data.index)
    else:
        df_flux_norm = flux_data
    return df_flux_norm

def load_metabolite_data(sample_id, preprocess=False, normalize=True):
    DATA_DIR = "/home/mongardi/spatial_biology/scFEA_data"
    if preprocess:
        file_name = sample_id + "_balance_prep.csv"
    else:
        file_name = sample_id + "_balance.csv"
    
    flux_data = pd.read_csv(os.path.join(DATA_DIR, file_name), index_col=0)

    if normalize:
        new_flux_data = MaxAbsScaler().fit_transform(flux_data)
        df_flux_norm = pd.DataFrame(new_flux_data, columns=flux_data.columns, index=flux_data.index)
    else:
        df_flux_norm = flux_data

    return df_flux_norm

def normalize_ST_data(arr):
    min_vals = np.nanmin(arr, axis=0)
    max_vals = np.nanmax(arr, axis=0)
    norm_arr = (arr - min_vals) / (max_vals - min_vals)
    col_mean = np.nanmean(norm_arr, axis=0)
    nan_mask = np.isnan(norm_arr)
    norm_arr[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return norm_arr


def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False, normalize=True):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    if normalize:
        print('Normalizing data')
        feat = normalize_ST_data(feat)
      
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a  
    if "flux_data" in adata.obsm.keys():
        adata.obsm["flux"] = adata.obsm["flux_data"].values
        flux_corr = permutation(adata.obsm["flux_data"].values)
        adata.obsm["flux_a"] = flux_corr
    if "met_data" in adata.obsm.keys():
        adata.obsm["met"] = adata.obsm["met_data"].values
        met_corr = permutation(adata.obsm["met_data"].values)
        adata.obsm["met_a"] = met_corr

def get_feature_only(adata, deconvolution=False, normalize=True):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    if normalize:
        print('Normalizing data')
        feat = normalize_ST_data(feat)
        # print(feat.max(axis=0)[0])
        # print(np.sum(np.isnan(feat)))
    adata.obsm['feat'] = feat 
    if "flux_data" in adata.obsm.keys():
        adata.obsm["flux"] = adata.obsm["flux_data"].values
    if "met_data" in adata.obsm.keys():
        adata.obsm["met"] = adata.obsm["met_data"].values

def add_contrastive_label(adata):
    # contrastive label
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL
    
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj, normalize=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    if normalize:
        return normalize_adj(adj)+np.eye(adj.shape[0])
    else:
        return adj + np.eye(adj.shape[0])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)    
    
def fix_seed(seed):
    #os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    

    