import numpy as np
import torch
import torch_geometric as tg

def make_undirected(mat):
    """Takes an input adjacency matrix and makes it undirected (symmetric).

    Parameters
    ----------
    mat: array
        Square adjacency matrix.

    Raises
    ------
    ValueError
        If input matrix is not square.

    Returns
    -------
    array
        Symmetric input matrix. If input matrix was unweighted, output is also unweighted.
        Otherwise, output matrix is average of corresponding connection strengths of input matrix.
    """
    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError('Adjacency matrix must be square.')

    sym = (mat + mat.transpose())/2
    if len(np.unique(mat)) == 2: #if graph was unweighted, return unweighted
        return np.ceil(sym) #otherwise return average
    return sym

# ANNABELLE
def knn_graph(mat,k=8,selfloops=False,symmetric=True):
    """Takes an input matrix and returns a k-Nearest Neighbour weighted adjacency matrix.

    Parameters
    ----------
    mat: array
        Input adjacency matrix, can be symmetric or not.
    k: int, default=8
        Number of neighbours.
    selfloops: bool, default=False
        Wether or not to keep selfloops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.
    
    Raises
    ------
    ValueError
        If input matrix is not square.
    ValueError
        If k not in range [1,n_nodes).
    
    Returns
    -------
    array
        Adjacency matrix of k-Nearest Neighbour graph.
    """
    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError('Adjacency matrix must be square.')

    dim = mat.shape[0]
    if (k<=0) or (dim <=k):
        raise ValueError('k must be in range [1,n_nodes)')

    m = np.abs(mat) # Look at connection strength only, not direction
    mask = np.zeros((dim,dim),dtype=bool)
    for i in range(dim):
        sorted_ind = m[:,i].argsort().tolist()
        neighbours = sorted_ind[-(k+1):] #self is considered
        mask[:,i][neighbours] = True
    adj = mat.copy() # Original connection strengths
    adj[~mask] = 0

    if not selfloops:
        np.fill_diagonal(adj,0)

    if symmetric:
        return make_undirected(adj)
    else:
        return adj

# LOIC
def knn_graph_quantile(mat, self_loops=False, k=8, symmetric=True):
    """Takes an input correlation matrix and returns a k-Nearest Neighbour weighted undirected adjacency matrix."""

    if not (mat.shape[0] == mat.shape[1]):
        raise ValueError("Adjacency matrix must be square.")
    dim = mat.shape[0]
    if (k <= 0) or (dim <= k):
        raise ValueError("k must be in range [1,n_nodes)")
    is_directed = not (mat == mat.transpose()).all()
    if is_directed:
        raise ValueError("Input adjacency matrix must be undirected (matrix symmetric)!")

    # absolute correlation
    mat = np.abs(mat)
    adj = np.copy(mat)
    # get NN thresholds from quantile
    quantile_h = np.quantile(mat, (dim - k - 1)/dim, axis=0)
    mask_not_neighbours = (mat < quantile_h[:, np.newaxis])
    adj[mask_not_neighbours] = 0
    if not self_loops:
        np.fill_diagonal(adj, 0)
    if symmetric:
        adj = make_undirected(adj)
    
    return adj

def make_group_graph(connectomes, k=8, self_loops=False, symmetric=True):
    """
    Parameters
    ----------
    connectomes: list of array
        List of connectomes in n_roi x n_roi format, connectomes must all be the same shape.
    k: int, default=8
        Number of neighbours.
    self_loops: bool, default=False
        Wether or not to keep self loops in graph, if set to False resulting adjacency matrix
        has zero along diagonal.
    symmetric: bool, default=True
        Wether or not to return a symmetric adjacency matrix. In cases where a node is in the neighbourhood
        of another node that is not its neighbour, the connection strength between the two will be halved.
    
    Raises
    ------
    ValueError
        If input connectomes are not square (only checks first).
    ValueError
        If k not in range [1,n_nodes).

    Returns
    -------
    graph
        Torch geometric graph object of k-Nearest Neighbours graph for the group average connectome.
    """
    if not (connectomes[0].shape[0] == connectomes[0].shape[1]):
        raise ValueError('Connectomes must be square.')
        
#     if not (connectomes.shape[0] == connectomes.shape[1]):
#         raise ValueError('Connectomes must be square.')

    # Group average connectome and nndirected 8 k-NN graph
    avg_conn = np.array(connectomes).mean(axis=0)
    avg_conn_k = knn_graph_quantile(avg_conn, k=k, self_loops=self_loops, symmetric=symmetric)

    # Format matrix into graph for torch_geometric
    adj_sparse = tg.utils.dense_to_sparse(torch.from_numpy(avg_conn_k))
    tg_graph = tg.data.Data(edge_index=adj_sparse[0], edge_attr=adj_sparse[1])

    return tg_graph

# if __name__ == "__main__":
#     import os
#     import matplotlib.pyplot as plt
#     import simexp_gcn
#     import simexp_gcn.data as data
#     import simexp_gcn.data.raw_data_loader
    
#     ts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data_shima", "processed", "hcptrt_MIST444", "timeseries")
#     pheno_path = os.path.join(os.path.dirname(__file__), "..", "..", "data_shima", "raw", "hcptrt", "phenotypic_data.tsv")
# #     DataLoad = simexp_gcn.data.raw_data_loader.RawDataLoader(ts_dir=ts_dir, conn_dir=conn_dir,  pheno_path=pheno_path)
#     conn_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data_shima", "processed", "hcptrt_MIST444", "connectomes")
#     conn_files = sorted(glob.glob(conn_dir + '/*.npy'))
        
#    # load connectomes
#     connectomes = []
#     for conn_file in conn_files:
#       connectomes += [np.load(conn_file)]
#     avg_conn = np.array(connectomes).mean(axis=0)    
    
#     import time
#     start = time.time()
#     for ii in range(50):
#         avg_conn8 = knn_graph(avg_conn, k=8)
#     print("{}s".format(time.time() - start))
#     start = time.time()
#     for ii in range(50):
#         avg_conn8_quantile = knn_graph_quantile(avg_conn, k=8)
#     print("{}s".format(time.time() - start))