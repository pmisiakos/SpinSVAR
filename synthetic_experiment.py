import experiments.utils
import experiments.data.utils
import experiments.evaluation.utils
import experiments.methods.utils
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import os


#plots
from experiments.plot_experiment import visualize

from threading import Thread
import functools

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


if __name__ == '__main__':
    parser, args = experiments.utils.get_args()
    print(vars(args))

    # naming the output files according to the experimental settings
    filename, label = experiments.utils.get_filename(parser, args)

    # make directory to put results
    if not os.path.exists("results/{}/".format(filename)):
        os.makedirs("results/{}/".format(filename))

    p = args.number_of_lags
    for n in args.samples:
        for d in args.nodes:
            for t in args.timesteps:
                experiments.utils.initialize_results_file(filename, n, t, d, args, label)

                current = {}
                avgT = {}

                for key in args.methods:
                    current[key] = []
                    avgT[key] = []

                r = 0
                for _ in tqdm(range(args.runs)):

                    # graph initialization
                    start = time.time()
                    
                    X, C_true, cond_num, B_true, W_true = experiments.data.utils.get_data(args, n, d, T=t, dataset="synthetic")
                    print("Total number of edges {}".format(np.sum(B_true)))
                    # X has shape n x T x d where n is the number of independent realizations, T the length of the time series and d the number of nodes in the dag
                    # B_true and W_true have shape d x (p + 1)d where p is the number of time-lags. They are expressed in the form B_true = [A, B_1, ..., B_p]

                    print("\n\nData generation process done. Time: {:.3f}\n\n".format(time.time() - start))

                    # normalizes or standardizes data if supposed to
                    X = experiments.data.utils.data_transform(X, args) 

                    # causal discovery algorithms
                    if not np.isnan(X).any() and experiments.utils.is_bounded(X):
                        for method in args.methods:
                            # try:
                            #     B_est, W_est, T = timeout(timeout=args.timeout)(experiments.methods.utils.execute_method)(X, method, args, n, d, t, dataset="time_series", ground_truth=B_true)
                            B_est, W_est, T = experiments.methods.utils.execute_method(X, method, args, n, d, t, dataset=args.dataset, ground_truth=B_true)
                            # except:
                            #     print("Time limit exceeded")
                            #     B_est, W_est, T = np.zeros((d, d * (args.number_of_lags + 1))), np.zeros((d, d * (args.number_of_lags + 1))), args.timeout

                            # to execute change assume unique to false
                            experiments.evaluation.utils.compute_metrics(method, current, filename, r, t, T, X, C_true, B_true, W_true, B_est, W_est, args)
                        r += 1

                # save average results in csv
                experiments.evaluation.utils.save_results(current, filename, args, r)
