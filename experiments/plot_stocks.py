import sys
import os
import pandas as pd

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
import pandas as pd
import experiments.data.utils
import experiments.evaluation.evaluation
from experiments.plot_experiment import visualize_stocks_adjacency, visualize_root_causes
from experiments import utils

if __name__ == '__main__':
    parser, args = utils.get_args()

    samples = args.samples #[200, 400, 600, 800, 1000] if(args.value == 'samples') else [400]
    variables =  args.nodes # if(args.value == 'variables') else [10] 
    noise = args.noise
    runs = args.runs 
    (a, b) = tuple(args.weight_bounds)
    k = args.edges
    methods = args.methods

    seed = 42
    parser, args = utils.get_args()
    filename_data = "X_{}".format("Close")
    method = "spinsvar"
    p = args.number_of_lags

    for t in [50]:
        X_log, X, D, _, _ = experiments.data.utils.get_data(args, 0, 0, dataset="S&P", filename_data=filename_data)
        samples, d = X.shape

        np.random.seed(seed)
        stocks = np.random.choice(d, 60, replace=False)
        stocks = np.sort(stocks)
        stocks = np.arange(d) # takes all nodes
        stocks = np.arange(60)

        np.random.seed(seed)
        dates = np.random.choice(samples - samples % t, 60, replace=False)
        dates = np.append(dates, -5)
        dates = np.sort(dates)


        for run in range(args.runs):
            W_est = pd.read_csv('results_UAI/{}/W_est_{}_{}_{}_run_{}.csv'.format("S&P500", method, t, "Close", run), header=None)
            W_est = W_est.to_numpy()
            W_est = np.where(np.abs(W_est) > args.omega, W_est, 0)
            d = W_est.shape[0]
            visualize_stocks_adjacency(W_est, p, t, args, method=method, ind=stocks, run=run)
            
            C_est, _, _, _ = experiments.evaluation.evaluation.rct_approximation(method, X_log, t, W_est)
            # C_est has shape n / T, d * T
            rc_threshold = 0.07
            C_est = np.where(np.abs(C_est) > rc_threshold, C_est, 0)
            visualize_root_causes(C_est, t, args, method=method, indx=dates, indy=stocks, run=run)

            # reshaping, so they both have size samples x d (nodes)
            n, dT = C_est.shape
            d = dT // t
            C_est = C_est.reshape(n * t, dT // t)
            X = X[:samples - samples % t, :]
            # C_est = X_log[:samples - samples % t, :]

            agreement_ratio, total = utils.root_causes_in_stocks(C_est, X, eps=0.5)
            print("Total number of significant (greater than {:.2f}) root causes is: {}".format(rc_threshold, total))
            print("Fraction of root causes in agreement with stocks monotonicity: {:.3f}".format(agreement_ratio))

            pos, neg, agreement_pos, agreement_neg, total = utils.root_causes_vs_dividends(C_est, X, D, t)
            print("There is a total of {} paid dividends in total: ".format(total))
            print("Out of those only {} agree with some positive significant (greater that {:.3f}) root cause".format(pos, rc_threshold))
            print("Out of those, there exist {} that agree with some significant positive change in the data".format(agreement_pos))
            print("Out of those only {} agree with some negative significant (less that -{:.3f}) root cause".format(neg, rc_threshold))
            print("Out of those, there exist {} that agree with some significant negative change in the data".format(agreement_neg))

            pos, neg, agreement_pos, agreement_neg, total = utils.dividends_largest_companies(C_est, X, D, t, 10)
            print("There is a total of {} paid dividends in total: ".format(total))
            print("Out of those only {} agree with some positive significant (greater that {:.3f}) root cause".format(pos, rc_threshold))
            print("Out of those, there exist {} that agree with some significant positive change in the data".format(agreement_pos))
            print("Out of those only {} agree with some negative significant (less that -{:.3f}) root cause".format(neg, rc_threshold))
            print("Out of those, there exist {} that agree with some significant negative change in the data".format(agreement_neg))

            utils.print_root_cause(C_est, "2024-02-01", "close_META")
            utils.print_root_cause(C_est, "2023-05-24", "close_NVDA")
            