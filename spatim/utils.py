import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import torch
import os 
import tqdm
from sklearn.metrics import silhouette_score, davies_bouldin_score
from joblib import Parallel, delayed


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    num_cluster = int(num_cluster) 
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    #res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), robjects.IntVector(range(int(num_cluster), num_cluster+1)),  modelNames)
    #num_cluster, modelNames)
    print(res)
    G = res.rx2('G')[0]
    loglik = res.rx2('loglik')[0]
    bic = res.rx2('bic')[0]
    prob = res.rx2('z')
    print(prob)
    print('Number of clusters: {}, Log-likelihood: {}, BIC: {}'.format(G, loglik, bic))
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, radius=50, key='emb', method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False, pca=True, n_cs=20):
    """\
    Spatial clustering based the learned representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    radius : int, optional
        The number of neighbors considered during refinement. The default is 50.
    key : string, optional
        The key of the learned representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1.
    end : float 
        The end value for searching. The default is 3.0.
    increment : float
        The step size to increase. The default is 0.01.   
    refinement : bool, optional
        Refine the predicted labels or not. The default is False.

    Returns
    -------
    None.

    """
    if pca:
        pca = PCA(n_components=n_cs, random_state=42) 
        embedding = pca.fit_transform(adata.obsm['emb'].copy())
        adata.obsm['emb_pca'] = embedding
    else:
        adata.obsm['emb_pca'] = adata.obsm['emb'].copy()
        
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    elif method == 'louvain':
       res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['louvain'] 

    elif method == 'gmm_sk':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42, init_params='kmeans')
        gmm.fit(adata.obsm['emb_pca'])
        labels = gmm.predict(adata.obsm['emb_pca'])
        adata.obs['domain'] = pd.Categorical(values=labels)

    elif method == 'gmm_pt':
        from torchgmm.bayes import GaussianMixture
        gmm = GaussianMixture(num_components=n_clusters, covariance_type='full')
        gmm.fit(adata.obsm['emb_pca'])
        labels = gmm.predict(adata.obsm['emb_pca'])
        print(labels)
        adata.obs['domain'] = labels
    
    elif method =='spectral':
        from sklearn.cluster import SpectralClustering
        sc_model = SpectralClustering(n_clusters=n_clusters, random_state=42)
        #affinity='nearest_neighbors'
        labels = sc_model.fit_predict(adata.obsm['emb_pca'])
        adata.obs['domain'] = pd.Categorical(values=labels)
        
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=2020, n_init="auto").fit(adata.obsm['emb_pca'])
        labels = kmeans.labels_
        adata.obs['domain'] = pd.Categorical(values=labels)

    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

def extract_top_value(map_matrix, retain_percent = 0.1): 
    '''\
    Filter out cells with low mapping probability

    Parameters
    ----------
    map_matrix : array
        Mapped matrix with m spots and n cells.
    retain_percent : float, optional
        The percentage of cells to retain. The default is 0.1.

    Returns
    -------
    output : array
        Filtered mapped matrix.

    '''

    #retain top 1% values for each spot
    top_k  = retain_percent * map_matrix.shape[1]
    output = map_matrix * (np.argsort(np.argsort(map_matrix)) >= map_matrix.shape[1] - top_k)
    
    return output 

def construct_cell_type_matrix(adata_sc):
    label = 'cell_type'
    n_type = len(list(adata_sc.obs[label].unique()))
    zeros = np.zeros([adata_sc.n_obs, n_type])
    cell_type = list(adata_sc.obs[label].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    mat = pd.DataFrame(zeros, index=adata_sc.obs_names, columns=cell_type)
    for cell in list(adata_sc.obs_names):
        ctype = adata_sc.obs.loc[cell, label]
        mat.loc[cell, str(ctype)] = 1
    #res = mat.sum()
    return mat

def project_cell_to_spot(adata, adata_sc, retain_percent=0.1):
    '''\
    Project cell types onto ST data using mapped matrix in adata.obsm

    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    adata_sc : anndata
        AnnData object of scRNA-seq reference data.
    retrain_percent: float    
        The percentage of cells to retain. The default is 0.1.
    Returns
    -------
    None.

    '''
    
    # read map matrix 
    map_matrix = adata.obsm['map_matrix']   # spot x cell
   
    # extract top-k values for each spot
    map_matrix = extract_top_value(map_matrix) # filtering by spot
    
    # construct cell type matrix
    matrix_cell_type = construct_cell_type_matrix(adata_sc)
    matrix_cell_type = matrix_cell_type.values
       
    # projection by spot-level
    matrix_projection = map_matrix.dot(matrix_cell_type)
   
    # rename cell types
    cell_type = list(adata_sc.obs['cell_type'].unique())
    cell_type = [str(s) for s in cell_type]
    cell_type.sort()
    #cell_type = [s.replace(' ', '_') for s in cell_type]
    df_projection = pd.DataFrame(matrix_projection, index=adata.obs_names, columns=cell_type)  # spot x cell type
    
    #normalize by row (spot)
    df_projection = df_projection.div(df_projection.sum(axis=1), axis=0).fillna(0)

    #add projection results to adata
    adata.obs[df_projection.columns] = df_projection
    
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    


def plot_UMAP(embeddings, labels=None, title="UMAP Projection", save_path=None, name=None):
  
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = reducer.fit_transform(embeddings)
    if labels is not None:
        unique_labels = sorted(set(labels))  
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}  
        labels = np.array([label_to_int[label] for label in labels])  
        cmap = plt.get_cmap("tab10", len(unique_labels)) 
    else:
        labels = None
        cmap = "Spectral"
    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, cmap=cmap, alpha=0.7, marker=".")
    
    if labels is not None:
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label("Classes")
        cbar.set_ticks(np.arange(len(unique_labels)))  # Set tick positions
        cbar.set_ticklabels(unique_labels)  # Set tick labels
    else:
        plt.colorbar(scatter, label="Class Labels" if labels is not None else "Density")


    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    if save_path and name:
        plt.savefig(os.path.join(save_path,name), dpi=300, bbox_inches="tight")
    
    else:
        plt.show()

def same_class_percentage(adj, labels):

    labels = np.asarray(labels)
    print(np.unique(labels))
    n = adj.shape[0]

    same_class_mask = (labels[:, None] == labels[None, :])
    same_class_weight = (adj * same_class_mask).sum(axis=1)

    total_weight = adj.sum(axis=1)


    pct_same = np.divide(
        same_class_weight,
        total_weight,
        out=np.zeros_like(same_class_weight, dtype=float),
        where=total_weight != 0
    )

    class_avg = {
        cls: pct_same[labels == cls].mean()
        for cls in np.unique(labels)
    }

    return pct_same, class_avg

def keep_same_class_edges(adj, labels):

    labels = np.asarray(labels)
    same_class_mask = (labels[:, None] == labels[None, :])
    adj_new = adj * same_class_mask

    return adj_new  

def fisher_z(r):
    return np.arctanh(np.clip(r, -0.999999, 0.999999))

def inverse_fisher_z(z):
    return np.tanh(z)

def weighted_average_correlations(corr_matrices):

    Z = np.stack([fisher_z(C) for C in corr_matrices], axis=0)

    mean_z = Z.mean(axis=0)
    var_z = Z.var(axis=0, ddof=1)

    weights = 1.0 / (var_z + 1e-8)

    weighted_z = np.sum(weights * Z, axis=0) / np.sum(weights, axis=0)

    return inverse_fisher_z(weighted_z)

def sample_level_correlation(
    X,
    n_genes=100,
    n_iter=10,
    random_state=0, 
    method = 'mean', 
    var_scale = 1.0
):
    """
    X: (n_samples, n_genes) expression matrix
    """
    rng = np.random.default_rng(random_state)
    n_samples, n_total_genes = X.shape

    corr_matrices = []

    for _ in tqdm.tqdm(range(n_iter)):
        gene_idx = rng.choice(n_total_genes, size=n_genes, replace=False)
        X_sub = X[:, gene_idx]

        # Sample–sample correlation
        C = np.corrcoef(X_sub, rowvar=True)
        mask = np.isnan(C)
        C[mask] = 0
        corr_matrices.append(C)

    Z = np.stack(corr_matrices, axis=0)
    if method == 'min':
        mean_z = Z.min(axis=0)
    else: 
        mean_z = Z.mean(axis=0)
        var_z = Z.var(axis=0, ddof=1)
        #confidence = 1.0 / (1.0 + var_z * var_scale)
        #mean_z = confidence * mean_z
        weights = 1.0 / (var_z*var_scale + 1e-8)
        #weighted_z 
        mean_z = np.sum(weights * Z, axis=0) / np.sum(weights, axis=0)

    final_corr = mean_z
    #final_corr = weighted_average_correlations(corr_matrices)
    return final_corr

def sample_negatives(adj_1, adj_2, corr_matrix, labels=None):

    # removing self-loops
    np.fill_diagonal(adj_1, 0)
    np.fill_diagonal(adj_2, 0)
    k = adj_2.sum()
    adj_1 = adj_1 - adj_2
    row_idx, col_idx = np.nonzero(adj_1)
    edge_index_1 = np.vstack((row_idx, col_idx))  
    values_1 = corr_matrix[edge_index_1[0], edge_index_1[1]]

    row_idx, col_idx = np.nonzero(adj_2)
    edge_index_2 = np.vstack((row_idx, col_idx))  
    values_2 = corr_matrix[edge_index_2[0], edge_index_2[1]]
    negative_samples = []
    for i in range(adj_1.shape[0]):
        V_n1 = values_1[edge_index_1[0] == i]
        V_n2 = values_2[edge_index_2[0] == i]
        #print(V_n1, V_n2)
        edge_to_remove = V_n1 > 1.5*np.max(V_n2) 
        #print(f"Spot {i}: removed {np.sum(edge_to_remove)} edges")
        negative_samples.append(edge_index_1[:, edge_index_1[0] == i][:, edge_to_remove])

    negative_samples = np.hstack(negative_samples)
    if labels is not None:
        percentages = compute_same_class_percentage(negative_samples, labels)
        print(percentages)
    print("Negative samples: ", negative_samples.shape[1])
    if negative_samples.shape[1] <= k:
        print(negative_samples.shape[1], k)
        return negative_samples

    else :
        print('Sampling negative edges...')
        rand_idx = np.random.choice(negative_samples.shape[1], size=k, replace=False)
        return negative_samples #[:, rand_idx]


def compute_same_class_percentage(adj, labels):
    
    from collections import defaultdict

    same_count = defaultdict(int)
    total_count = defaultdict(int)
    print(adj.shape)
    print(adj[0].shape)
    for i in range(adj.shape[1]):
        u = adj[0, i]
        v = adj[1, i]
        total_count[u] += 1
        if labels.iloc[u] == labels.iloc[v]:
            same_count[u] += 1

    # percentage per node
    percentage = {
        u: (same_count[u] / total_count[u]) * 100
        for u in total_count
    }

    return percentage

def lowest_correlated_samples(C, k):
    n_samples = C.shape[0]
    indices = np.zeros((n_samples, k), dtype=int)

    for i in range(n_samples):
        corr_row = C[i].copy()
        corr_row[i] = np.inf

        indices[i] = np.argsort(corr_row)[:k]

    sources = np.repeat(np.arange(n_samples), k)
    targets = indices.reshape(-1)
    edges = np.vstack((sources, targets))
    print("Lowest correlated samples: ", edges.shape[1])
    return edges


def run_mclust_R(adata, num_cluster, random_seed=2020, modelNames='EEE', used_obsm='emb_pca'):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    num_cluster = int(num_cluster) 
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])
    mclust_res = mclust_res.astype('int')
    SC = silhouette_score(adata.obsm['emb'], mclust_res)
    DB = davies_bouldin_score(adata.obsm['emb'], mclust_res) 
    return num_cluster, SC, DB, mclust_res 


def stable_clustering(adata, n_clusters_max=20,key='emb', method='mclust', pca=True, n_cs=20, n_runs=10, refit=False):

    if pca:
        pca = PCA(n_components=n_cs, random_state=42) 
        embedding = pca.fit_transform(adata.obsm[key].copy())
        adata.obsm['emb_pca'] = embedding
    else:
        adata.obsm['emb_pca'] = adata.obsm[key].copy()
    
    silh_dict = {i : [] for i in range(5, n_clusters_max+1)}
    db_dict = {i : [] for i in range(5, n_clusters_max+1)}
    for i in range(n_runs):
        for j in range(5, n_clusters_max+1):
       
            adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=j, seed=2020+i)
            SC = silhouette_score(adata.obsm['emb_pca'], adata.obs['mclust'])
            DB = davies_bouldin_score(adata.obsm['emb_pca'], adata.obs['mclust'])   
            silh_dict[j].append(SC)
            db_dict[j].append(DB)
    
    silh_avg = {k: np.mean(v) for k, v in silh_dict.items()}
    db_avg = {k: np.mean(v) for k, v in db_dict.items()}
    print("Average Silhouette Scores:", silh_avg)
    print("Average Davies-Bouldin Scores:", db_avg)
    best_k_sc = max(silh_avg, key=silh_avg.get)
    best_k_db = min(db_avg, key=db_avg.get)
    print(f"Best k by Silhouette Score: {best_k_sc}")
    print(f"Best k by Davies-Bouldin Score: {best_k_db}")

    # if refit:
    #     for i in range(n_runs):
    #         adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=best_k_sc, seed=2020+i)

    return best_k_sc, best_k_db


def stable_clustering_parallel(
    adata,
    n_clusters_min=10,
    n_clusters_max=20,
    key='emb',
    pca=True,
    n_cs=20,
    n_runs=10,
    n_jobs=-1,
    refit=False
):

    if pca:
        pca = PCA(n_components=n_cs, random_state=42) 
        embedding = pca.fit_transform(adata.obsm[key].copy())
        adata.obsm['emb_pca'] = embedding
    else:
        adata.obsm['emb_pca'] = adata.obsm['emb'].copy()
        #embedding = adata.obsm[key].copy()

    ks = range(n_clusters_min, n_clusters_max + 1)
    tasks = [
        (k, 2020 + i)
        for i in range(n_runs)
        for k in ks
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_mclust_R)(adata, k, seed)
        for k, seed in tasks
    )

    silh_dict = {k: [] for k in ks}
    db_dict = {k: [] for k in ks}

    for k, sc, db, _ in results:

        silh_dict[k].append(sc)
        db_dict[k].append(db)

    silh_avg = {k: np.mean(v) for k, v in silh_dict.items()}
    db_avg = {k: np.mean(v) for k, v in db_dict.items()}
    ratio = {k: silh_avg[k]/db_avg[k] for k in ks}
    print("Average Silhouette Scores:", silh_avg)
    print("Average Davies-Bouldin Scores:", db_avg)
    best_k_sc = max(silh_avg, key=silh_avg.get)
    best_k_db = min(db_avg, key=db_avg.get)
    best_k_ratio = max(ratio, key=ratio.get)
    print(f"Best k by Silhouette Score: {best_k_sc}")
    print(f"Best k by Davies-Bouldin Score: {best_k_db}")
    print(f"Best k by Silhouette/DB Ratio: {best_k_ratio}")
    if refit:
        clusterings = []
        tasks = [
        (best_k_ratio, 2020 + i)
        for i in range(n_runs)]

        results = Parallel(n_jobs=n_jobs)(
        delayed(run_mclust_R)(adata, k, seed)
        for k, seed in tasks)

        for  _, _, _, clustering in results:
            clusterings.append(clustering)
        return best_k_sc, best_k_db, best_k_ratio, clusterings
    return best_k_sc, best_k_db, best_k_ratio


def build_coclustering_matrix(labels_list, normalize=True):
    print("Building coclustering matrix...")

    K = len(labels_list)
    N = len(labels_list[0])
    M = np.zeros((N, N), dtype=np.float32)
    for labels in labels_list:
        labels = np.asarray(labels)
        M += (labels[:, None] == labels[None, :])
    if normalize:
        M /= K
        
    return M

def get_final_clustering(coclust_matrix, n_clusters):

    from sklearn.cluster import AgglomerativeClustering

    clustering_model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average',
        #distance_threshold=1.0 - threshold
    )

    distance_matrix = 1.0 - coclust_matrix
    labels = clustering_model.fit_predict(distance_matrix)
    return labels


def compute_similarity_torch_batch(xi, X, gene_idx):

    xi_sub = xi[gene_idx]
    X_sub = X[:, gene_idx].permute(1, 0, 2)
    xi_sub = F.normalize(xi_sub, dim=1)        # (n_iter, k)
    X_sub = F.normalize(X_sub, dim=2)           # (n_iter, n_spots, k)

    # Batched dot product
    sim = torch.einsum("ik,ijk->ij", xi_sub, X_sub)
    return sim


def stochastic_similarity_matrix_torch_adj_v2(
    X,
    adj,
    k=50,
    n_iter=20,
    gene_weights_init=None,
    random_state=0,
    scaling=True,
    drop_housekeeping=True,
    house_keeping_genes=None,
    device="cpu"
):
    torch.manual_seed(random_state)

    X = torch.as_tensor(X, device=device, dtype=torch.float32)
    adj = torch.as_tensor(adj, device=device)

    n_spots, n_genes = X.shape

  
    S_accum = torch.zeros((n_spots, n_spots), device=device)
    counts = torch.zeros(n_spots, device=device)
    var = torch.zeros((n_spots, n_spots), device=device)

    if house_keeping_genes is not None:
        hk = torch.as_tensor(house_keeping_genes, device=device)

    for i in tqdm.tqdm(range(n_spots)):
        xi = X[i]

        neighbors = adj[i].nonzero(as_tuple=True)[0]
       
        if neighbors.numel() == 0:
            continue

        expressed = (xi > 0).nonzero(as_tuple=True)[0]
      
        #if expressed
        if drop_housekeeping and house_keeping_genes is not None:
            expressed = expressed[~torch.isin(expressed, hk)]

        if expressed.numel() < 2:
            continue

        weights = (
            gene_weights_init[expressed]
            if gene_weights_init is not None
            else xi[expressed]
        )
        weights = torch.as_tensor(weights, device=device)

        sims = torch.zeros((n_iter, neighbors.numel()), device=device)

        X_neighbors = X[neighbors]
        if weights.numel() <= k:

            weights = xi.clone()
            gene_samples = torch.multinomial(
                        weights,
                        num_samples=min(k * n_iter, weights.numel() * n_iter),
                        replacement=True
                    ).view(n_iter, k)

            #gene_samples = torch.arange(weights.numel(), device=device).repeat(n_iter, 1)
            gene_idx = gene_samples
        else:
            gene_samples = torch.multinomial(
                        weights,
                        num_samples=min(k * n_iter, weights.numel() * n_iter),
                        replacement=True
                    ).view(n_iter, k)
            gene_idx = expressed[gene_samples] 
        sims = compute_similarity_torch_batch(xi,  X_neighbors, gene_idx)
        S_accum[i, neighbors] += sims.sum(dim=0)
        counts[i] += n_iter
        var[i, neighbors] = sims.max(dim=0).values - sims.min(dim=0).values

    if scaling:
        S = S_accum / (counts[:, None] * (1.0 + var))
    else:
        S = S_accum / counts[:, None]

    S.fill_diagonal_(1.0)
    #fill spots with no neighbors with cosine similarity to all spots
    S[S.sum(axis=1)==1] = adj[S.sum(axis=1)==1].float()
    return S.cpu().numpy()