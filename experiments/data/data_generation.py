import numpy as np
import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def simulate_time_unrolled_dag(d, s0, graph_type, number_of_lags=0, average_degrees_per_lagged_node=[None, None]):
    """Simulate random time unrolled DAG.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges for instantaneous adjacency matrix
        graph_type (str): ER, SF, BP
        number_of_lags: number of additional adjecency matrices for non-instantaneous dependencies
        average_degrees_per_lagged_node: degrees of the adjacency matrices that represent non-instantaneous dependencies

    Returns:
        A (list of np.ndarray): list of (number_of_lags + 1) elements where each is a [d, d] binary adj matrix.
                                The first adjacency matrix represents instantaneous dependencies B_0
                                and the rest number_of_lags the lagged dependencies B_1,...,B_k.

    Function adapted from notears repository:
    https://github.com/xunzheng/notears/blob/master/notears/utils.py
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    W = [] # initializing window graph
    for t in range(number_of_lags + 1):
        s0 = average_degrees_per_lagged_node[t - 1] * d if t > 0 else s0

        if graph_type == 'ER':
            # Erdos-Renyi
            G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
            B_und = _graph_to_adjmat(G_und)
            B = _random_acyclic_orientation(B_und)
        elif graph_type == 'SF':
            # Scale-free, Barabasi-Albert
            G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
            B = _graph_to_adjmat(G)
        elif graph_type == 'BP':
            # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
            top = int(0.2 * d)
            G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
            B = _graph_to_adjmat(G)
        else:
            raise ValueError('unknown graph type')
        
        B_perm = _random_permutation(B)
        assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
        W.append(B_perm)

    return W


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Assigning random weights to an adjacency matrix.

    Args:
        B (np.ndarray): [d, d] binary adj matrix
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def sparse_input_sem(W_full, T, n=1, sparsity=0.3, std=0.01, noise_type='gauss', sparsity_type="uniform"):
    """
        --- Optimal implementation -----
        W_full : list of adjacencies (length = p + 1)
        W_full = [A, B_1, B_2, ..., B_p]
        T: number of desired timesteps
        n: number of sequences to produce
    """

    #number of nodes
    d = W_full[0].shape[0]
    p = len(W_full)  - 1

    # matrices
    A = W_full[0]
    I = np.eye(d)
    A = csc_matrix(I - A.T)
    
    B = np.concatenate(W_full[1:][::-1], axis=0) # B = [B_p
                                                 #      B_{p-1}
                                                 #      ...
                                                 #      B_2
                                                 #      B_1]

#--sparsity 0.0 --noise laplace --noise_std 0.033  --dataset laplace

    # initializing the sparse input
    if sparsity_type == "laplace":
        C = np.random.laplace(loc=0, scale=0.033, size=(n, d * T))
        Nf = np.zeros((n, d * T))
        
    elif sparsity_type == "uniform":
        pos = np.random.choice([0, 1], size=(n, d * T), p=[1 - sparsity, sparsity]) 
        sign = np.random.choice([-1, 1], size=(n, d * T)) 
        C = pos * np.random.uniform(0.1, 1, size=(n, d * T)) * sign

    elif sparsity_type == "gauss":
        pos = np.random.choice([0, 1], size=(n, d * T), p=[1 - sparsity, sparsity]) 
        sign = np.random.choice([-1, 1], size=(n, d * T)) 
        C = pos * np.random.normal(scale=0.1, loc=0.5, size=(n, d * T)) * sign

    # computing matrix of independent noises
    if std==0 or sparsity_type == "laplace":
        Nf = np.zeros((n, d * T))
    elif noise_type == 'gauss':
        Nf = np.random.normal(scale=std, size=(n, d * T))
    elif noise_type == 'laplace':
        Nf = np.random.laplace(loc=0, scale=std, size=(n, d * T))
    else: # considering gumbel case
        noise_scale = np.sqrt(6) * std / np.pi
        Nf = np.random.gumbel(scale=noise_scale, size=(n, d * T))

    # adding noise to the root causes
    C_noisy = C + Nf
    X = np.zeros(C_noisy.shape)

    X[:, :d] = C_noisy[:, :d]
    for t in range(1, T):
        if t < p:
            Y = X[:, 0 : t * d] @ B[- (t * d):, :] + C_noisy[:, t * d : (t + 1) * d] # y = [x(t-p) ... x(t-1)] B + c[t]
        else:
            Y = X[:, (t - p) * d : t * d] @ B + C_noisy[:, t * d : (t + 1) * d] # y = [x(t-p) ... x(t-1)] B + c[t]


        X[:, t * d : (t + 1) * d] = spsolve(A, Y.T).T # x[t] = x[t]A + y

    X = X.reshape(n, T, d)
    C = C.reshape(n, T, d)
    Nf = Nf.reshape(n, T, d)

    return X, C #, C + Nf



if __name__ == "__main__":
    # testing the above functionalities
    import numpy as np

    a = np.array([0])
    b = np.array([1])

    W_full = [a,b,a,b]

    X, C = sparse_input_sem(W_full, 10, n=1, sparsity=0.1, std=0)
    print(X, C)

