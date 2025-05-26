import numpy as np
import experiments.evaluation.evaluation 
import experiments.utils
import pandas as pd
import cdt 
import sklearn.metrics
# cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.2.1/bin/Rscript'

def compute_metrics(method, current, filename, r, t, T, X, C_true, B_true, W_true, B_est, W_est, args):
    C_est, c_nmse, c_num, c_shd = experiments.evaluation.evaluation.input_approximation(method, X, t, W_est, C_true)
    nmse = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)
    shd = 0
    E = args.edges * d + 2 * d * args.number_of_lags
    d = X.shape[-1]
    for i in range(args.number_of_lags + 1):
        shd += cdt.metrics.SHD(B_true[:, i * d : (i + 1) * d], B_est[:, i * d : (i + 1) * d], double_for_anticausal=False)
    nSHD = shd / E

    # acc = sklearn.metrics.accuracy_score(B_true.flatten(), B_est.flatten()) # tp + tn / (p + n)
    tpr = (B_true * B_est).sum() / B_true.sum() # tp / (tp + fn)
    nnz = np.sum(B_est)
    prec = sklearn.metrics.precision_score(B_true.flatten(), B_est.flatten()) # tp / (tp + fp) (1-FDR)
    rec = sklearn.metrics.recall_score(B_true.flatten(), B_est.flatten()) # tp / (tp + fn)
    f1 = sklearn.metrics.f1_score(B_true.flatten(), B_est.flatten()) # harmonic mean
    auroc = sklearn.metrics.roc_auc_score(B_true.flatten(), B_est.flatten()) # area under ROC curve
    sid = 0 # cdt.metrics.SID(B_true, B_est) if d < 200 else 0 # sid is too expensive

    results = [nSHD, shd, tpr, nnz, prec, rec, f1, auroc, nmse, T, sid, c_nmse, c_num, c_shd]
    current[method].append(results)
    print_results(results, filename, method)

    # # looking at weights
    # if d > 100:
    #     df = pd.DataFrame(W_est)
    #     df.to_csv('results/W_est_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None, index=False)
    #     df = pd.DataFrame(W_true)
    #     df.to_csv('results/W_true_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None, index=False)


def compute_varsortability(avg_varsortability, filename, args):
    with open('results/{}.csv'.format(filename), 'a') as f:
        # computing varsortability of dataset 
        avg_varsortability = avg_varsortability / args.runs
        print("Avg Varsortability, {:.3f}".format(avg_varsortability))
        f.write("Avg Varsortability, {:.3f}\n".format(avg_varsortability))
    f.close()

def cond_num(avg_cond_num, filename, args):
    with open('results/{}.csv'.format(filename), 'a') as f:
        avg_cond_num = avg_cond_num / args.runs
        print('Avg cond num of (I + transclos(W)) is {:.3f}'.format(avg_cond_num))
        f.write('Avg cond num of (I + transclos(W)) is {:.3f}\n'.format(avg_cond_num))
    f.close()

def save_results(current, filename, args, r):      
    with open('results/AVG_{}.csv'.format(filename), 'a') as f:
        # Log results
        avg = {}
        std = {}

        f.write("Total executions were {}\n".format(r))
        print("Total executions were {}".format(r))
        
        for method in args.methods:
            avg[method] = np.mean(current[method], axis=0)
            std[method] = np.std(current[method], axis=0)
            
            f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, avg[method][0], avg[method][1], avg[method][2], avg[method][3], avg[method][4], avg[method][5], avg[method][6], avg[method][7], avg[method][8], avg[method][9], avg[method][10], avg[method][11], avg[method][12], avg[method][13]))
            f.write("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, std[method][0], std[method][1], std[method][2], std[method][3], std[method][4], std[method][5], std[method][6], std[method][7], std[method][8], std[method][9], std[method][10], std[method][11], std[method][12], std[method][13]))
            print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, avg[method][0], avg[method][1], avg[method][2], avg[method][3], avg[method][4], avg[method][5], avg[method][6], avg[method][7], avg[method][8], avg[method][9], avg[method][10], avg[method][11], avg[method][12], avg[method][13]))
            print("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, std[method][0], std[method][1], std[method][2], std[method][3], std[method][4], std[method][5], std[method][6], std[method][6], std[method][8], std[method][9], std[method][10], std[method][11], std[method][12], std[method][13]))
    f.close()

def print_results(results, filename, method, search_params=None):  
    if search_params is not None:   
        with open('results/{}.csv'.format(filename), 'a') as f: 
            f.write("Method is {} with params \n".format(method))
            for key in search_params.keys():
                f.write("{} has value {}\n".format(key, search_params[key]))
            f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10], results[11], results[12], results[13]))
            
            print("Method is {} with params".format(method))
            for key in search_params.keys():
                print("{} has value {}".format(key, search_params[key]))
            print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10], results[11], results[12], results[13]))
        f.close()

    else:
        with open('results/{}.csv'.format(filename), 'a') as f: 
            f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10], results[11], results[12], results[13]))
            print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], results[8], results[9], results[10], results[11], results[12], results[13]))
        f.close()

def print_best_params(filename, method, search_params):   
    with open('results/{}.csv'.format(filename), 'a') as f: 
        f.write("BEST PARAMS for {}\n".format(method))
        for key in search_params.keys():
            f.write("{} has value {}\n".format(key, search_params[key]))
        
        print("BEST PARAMS for {}".format(method))
        for key in search_params.keys():
            print("{} has value {}".format(key, search_params[key]))

    f.close()