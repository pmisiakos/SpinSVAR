import numpy as np
import igraph as ig
import argparse
import random
import pandas as pd
import torch 
import subprocess
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', default='gauss', type=str, help='Choices none, gauss, gumbel, uniform')
    parser.add_argument('--noise_std', default=0.01, type=float, help='Noise magnitude')
    parser.add_argument('--sparsity', default=0.05, type=float, help='Probability of data being nonzero at vertex v')
    parser.add_argument('--sparsity_type', default="uniform", type=float, help='What type of sparsity distribution to consider. Default is uniform which assigns a uniform weight to the non-zero input entries (chosen at random as bernoulli). Other choices laplacian (without bernoulli) and Gaussian (with bernoulli).')
    
    parser.add_argument('--dataset', default='time_series', type=str, help='time_series, finance, S&P 500, dream3')
    parser.add_argument('--weight_bounds', default=[0.1, 0.5], nargs='+', type=float, help='initialization of weighted adjacency matrix')
    parser.add_argument('--samples', default=[10], nargs='+', type=int, help='number of samples')
    parser.add_argument('--timesteps', default=[1000], nargs='+', type=int, help="For how many timesteps to generate time series data")
    parser.add_argument('--number_of_lags', default=2, type=int, help="How far in the past can a node affect the current one. 0 means only current nodes can affect")
    parser.add_argument('--graph_type', default='ER', type=str, help='Choices ER (Erdös-Renyi), SF (Scale Free)')
    parser.add_argument('--nodes', default=[20], nargs='+', type=int, help='number of graph vertices to consider')# [5, 10, 15, 20, 25]
    parser.add_argument('--edges', default=5, type=int, help='graph has k * d edges')
    parser.add_argument('--transformation', default='None', type=str, help='Whether to normalize/standardize the given signals')

    parser.add_argument('--methods', default=["spinsvar", "sparserc", "varlingam", "d_varlingam", "dynotears", "nts-notears", "tsfci", "pcmci", "TCDF"], nargs='+', type=str, help='methods to compare') 
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--timeout', default=600, type=int)

    parser.add_argument('--algo_lags', default=2, type=int, help="How far in the past assumes algo that can a node affect the current one. 0 means only current nodes can affect")
    parser.add_argument('--omega', default=0.09, type=float, help='Thresholding the output matrix')
    parser.add_argument('--lambda1', default=0.001, type=float, help="Sparsity regularizer coefficient")
    parser.add_argument('--lambda2', default=1, type=float, help="Acyclicity regularizer coefficient")
    
    parser.add_argument('--alpha', default=1, type=float, help='')
    parser.add_argument('--beta', default=1, type=float, help='')
    
    parser.add_argument('--table', default='TPR', type=str, help='Choices TPR, SHD')
    parser.add_argument('--legend', default='False', type=str, help='Whether to plot the legend only')
    parser.add_argument('--rotate', default='False', type=str, help='Whether rotate xlabels')
    args = parser.parse_args()

    return parser, args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode('utf-8')
        print("CUDA Version found:\n", nvcc_version)
        return True
    except Exception as e:
        print("CUDA not found or nvcc not in PATH:", e)
        return False


def get_filename(parser, args):
    # naming the output files according to the experimental settings 
    dic = vars(args)
    filename = ''
    label = ''

    if args.dataset in ['finance', 'dream3']:
        return '{}'.format(args.dataset), ''
    for key in dic.keys():
        if(key not in ['methods', 'nodes', 'variables', 'legend', 'rotate'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])

        label += '{} = {}, '.format(key, dic[key])
    filename = filename if len(filename) > 0 else 'default'
    filename = re.sub(r"[\[\]\s]", "_", filename) # replacing all brackets and whitespaces with underscore -> arxiv compatibility
    return filename, label

def edges_to_adjacency(filename, d, time_lag):
    "reads a csv where each row has format a, b, time_lag and returns the corresponding adjacency matrix."
    "if the max time lag is more than 1 then it returns as many adjacency matrices as the time lag."

    df = pd.read_csv(filename, header=None)
    edges = df.to_numpy()
    
    adjacency = np.zeros((d, d * (time_lag + 1)))
    
    for a, b, time_lag in edges:
        adjacency[a, b + time_lag * d ] = 1 
    
    return adjacency


def network_to_numpy(g_learnt, d, number_of_lags):
    """
    Translating causalnex output format to adjacency matrix format.
    An edge of the form i_lagj, k_lag0 means that node k at time t is affected by node i at time t-j.
    Thus the entry (i, k) is non-zero (=1) at the matrix B_j (where B_0 = A).
    """
    B_est = np.zeros((d, (number_of_lags + 1) * d))
    W_est = np.zeros((d, (number_of_lags + 1) * d))
    # for i in range(number_of_lags + 1):
    #     B_est.append(np.zeros((d, d)))
    #     W_est.append(np.zeros((d, d)))

    for (a, b, w) in g_learnt.edges.data("weight"):
        parent, parent_lag = [int(x) for x in a.split("_lag")]
        child, _ = [int(x) for x in b.split("_lag")]

        # B_est[parent_lag][parent, child] = 1
        B_est[parent, parent_lag * d + child] = 1
        W_est[parent, parent_lag * d + child] = w
        # W_est[parent_lag][parent, child] = w

    return B_est, W_est

def block_toeplitz(W_full, T):
    """
        W_full : list of adjacencies (length = p + 1)
        T: number of desired timesteps
    """
    if isinstance(W_full, np.ndarray): 
        #number of nodes
        d = W_full.shape[0]
        p = W_full.shape[1] // d - 1 # number of lags
        W_list = [W_full[:d, i * d : (i + 1) * d] for i in range(p + 1)] # assumes that the first matrix corresponds to intra-slice dependencies
    elif isinstance(W_full, list):
        W_list = W_full
        d = W_list[0].shape[0]
        p = len(W_list) - 1
    
    I = np.eye(T)
    I_shift = np.roll(I, 1, 1)
    I_shift[-1:, 0] = 0

    # computing 
    # |W 0 0 0 |
    # |0 W 0 0 |
    # |0 0 W 0 |
    # |0 0 0 W | 
    A = np.kron(I, W_list[0])

    # computing 
    # |0 W 0 0 |
    # |0 0 W 0 |
    # |0 0 0 W |
    # |0 0 0 0 |
    I_i = I_shift
    for i in range(p):
        A += np.kron(I_i, W_list[i + 1])
        I_i = I_i @ I_shift

    # result 
    # |W_0 W_1 W_2 0   |
    # |0   W_0 W_1 W_2 |
    # |0   0   W_0 W_1 |
    # |0   0   0   W_0 | 
    return A 


def is_bounded(X):
    '''
    checks if the data have  "normal" values
    '''
    n, T, d = X.shape
    # print("Warning data has mean {:.4f}".format(np.abs(X).mean()))
    return np.abs(X).mean() < n * d * T * 10000

def initialize_results_file(filename, n, t, d, args, label):
    print('samples = {}, timesteps = {}, nodes = {}, edges = {}'.format(n, t, d, args.edges * d + 2 * d * args.number_of_lags))

    # file for average results 
    with open('results/AVG_{}.csv'.format(filename), 'a') as f:
        f.write('{}\n'.format(label))
        f.write('samples = {}, timesteps = {}, nodes = {}, edges = {}\n'.format(n, t, d, args.edges * d + 2 * d * args.number_of_lags))
    f.close()

    # file for all results
    with open('results/{}.csv'.format(filename), 'a') as f:
        f.write('{}\n'.format(label))
        f.write('samples = {}, timesteps = {}, nodes = {}, edges = {}\n'.format(n, t, d, args.edges * d + 2 * d * args.number_of_lags))
    f.close()


def X_past(X, k, device="cpu"):
    '''
    for i = 1,..., N sample X[i] = [x_0
                                    x_1
                                    ...
                                    x_{T-1} ]
    We create X_past[i] = [0, ..., 0, 0, x_0
                           0, ..., 0, x_1, x_0
                            ...
                           x_{T-k-1}, ..., x_{T-3}, x_{T-2}, x_{T-1}]
        ] 
    '''
    n, T, d = X.shape
    X = torch.tensor(X).reshape(n, d * T)
    X_past = torch.zeros((n, T, (k + 1) * d), device=device)
    for t in range(T):
        if t < k :
            X_past[:, t, :] = torch.cat([torch.zeros((n, (k - t) * d), device=device), X[:, :(t + 1) * d]], dim=1)
        else:
            X_past[:, t, :] = X[:, (t - k) * d: (t + 1) * d] 
        
    print(X_past.shape)

    return X_past

def overlapping_chunks(X, k, T=None, real=False):
    '''
    Takes time-series graph data X of shape n x T x d and 
    returns chunked data of shape X_chunk n(T - k + 1) x kd

    if real is True then we refer to a real dataset of shape n x Td
    '''
    if not real:
        n, T, d = X.shape
        X = X.reshape(n, d * T)
    else:
        n, DT = X.shape
        d = int(DT / T)

    X_chunk = np.concatenate([np.concatenate([[X[i, j * d : (j + k) * d]] for j in range(T - k + 1)], axis=0) for i in range(n)], axis = 0)

    print(X_chunk.shape)

    return X_chunk\
    


if __name__ == "__main__":
    # testing the above functionalities
    import numpy as np

    a = np.array([0])
    b = np.array([1])

    W_full = [a,b,a,b]

    W = block_toeplitz(W_full, 10)
    print(W)

    W_full = np.array([[1,2,3,4],[5,6,7,8]]) 
    W = block_toeplitz(W_full, 7)
    print(W)
