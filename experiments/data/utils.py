import experiments.data.data_generation
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np

def get_data(args, n, d, T=10, dataset="synthetic", dataset_id=0, filename_data=None, filename_gt=None):
    if (dataset in ["time_series", "time_series_laplace", "laplace", "synthetic"]):
        (a, b) = tuple(args.weight_bounds)
        k = args.edges

        # Initiating the random DAG
        average_degrees_per_lagged_node = [2 for _ in range(args.number_of_lags)]
        B_true = experiments.data.data_generation.simulate_time_unrolled_dag(d, k * d, args.graph_type, args.number_of_lags, average_degrees_per_lagged_node) # random graph simulation with avg degree = k

        # Initializing weights on the adjacency matrix
        W_true = experiments.data.data_generation.simulate_parameter(np.array(B_true), w_ranges=((-b, -a), (a, b))) # sampling uniformly the weights            
        W_true = list(W_true)

        X, C_true = experiments.data.data_generation.sparse_input_sem(W_true, T, n=n, sparsity=args.sparsity, std=args.noise_std,
                                    noise_type=args.noise, sparsity_type=args.sparsity_type)

        cond_num = 0
        W_true = np.concatenate(W_true, axis=1)
        B_true = np.concatenate(B_true, axis=1)

        return X, C_true, cond_num, B_true, W_true


def data_transform(X, args):
    # applying transformation to data (or not)
    if (args.transformation == 'norm'):
        X = normalize(X)
    elif (args.transformation == 'stand'):
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    return X
    