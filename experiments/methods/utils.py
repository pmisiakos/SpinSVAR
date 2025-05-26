import time 
import numpy as np
import torch 
import experiments.utils

# sparserc
from sparserc.spinsvar import spinsvar_solver


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def execute_method(X, method, args, n, d, t, dataset="time_series", search_params=None, ground_truth=None):
    # X has shape n x t x d where 
    # n: number of independent instantiations
    # t: length of the time-series
    # d: number of nodes  
    # search_params: used for hyperparameter search, otherwise we used the best-performing hyperparameters
    # ground truth: for algorithms that output edges with ambiguity we need the ground truth to allow the method get the correct result

    if dataset == 'time_series':
        best_params = {
            "spinsvar" : {"lambda1" : 0.0001, "lambda2" :  0.1, "omega": 0.09},
        }
    elif dataset == 'laplace':
        best_params = {
            "spinsvar" : {"lambda1" : 0.0005, "lambda2" :  0.5, "omega": 0.09},
        }
    elif dataset == 'finance':
        best_params = {
            "spinsvar" : {"lambda1" : 0.01, "lambda2" :  1, "omega": 0.5}, 
        }
    elif dataset == 'dream3':
        best_params = {
            "spinsvar" : {"lambda1" : 0.001, "lambda2" :  10, "omega": 0.2}, # {"lambda1" : 0.0001, "lambda2" :  1, "omega": 0.1}, 
        }
    else: #dataset == 'S&P':
        best_params = {
            "spinsvar" : {"lambda1" : 0.01, "lambda2" :  1, "omega": 0.5}, 
        }

    if search_params is None and method in best_params.keys():
        params = best_params[method]
    else:
        params = search_params

    if method == 'spinsvar':
        start = time.time()
        if (dataset in ["time_series", "laplace"]):
            W = spinsvar_solver(X, lambda1=params["lambda1"], lambda2=params["lambda2"], time_lag=args.algo_lags, epochs=args.sparserc_epochs, omega=args.omega, T=t)
            L = min(args.number_of_lags, args.algo_lags)
            W_est = W[:d, :(L + 1) * d]
            if args.number_of_lags > args.algo_lags:
                W_est = np.concatenate([W_est, np.zeros((d, d * (args.number_of_lags - args.algo_lags)))], axis=1)

            print("Method is spinsvar with params l1 {} and l2 {}".format(params["lambda1"], params["lambda2"]))
        
        elif(dataset in ["thames", "us_temps", "stocks", "finance", "fMRI", "S&P", "swiss_temps", "dream3"]):
            a, _ = X.shape
            X = X[:int(a / t) * t, :]
            X = X.reshape((int(a / t), t, d))
            if dataset in ["finance", "S&P"]:
                W = spinsvar_solver(X, lambda1=params["lambda1"], lambda2=params["lambda2"], time_lag=args.number_of_lags, epochs=args.sparserc_epochs, omega=params["omega"], T=t)
            elif dataset == "dream3":
                W = spinsvar_solver(X, lambda1=params["lambda1"], lambda2=params["lambda2"], time_lag=args.algo_lags, epochs=args.sparserc_epochs, omega=params["omega"], T=t)
            W_est = W[:d, :(args.algo_lags + 1) * d]

        print(" Time for spinsvar was {:.3f}".format(time.time() - start))
        T = time.time() - start
        B_est = W_est != 0

    else: 
        print("method not implemented")
    
    return B_est, W_est, T
    