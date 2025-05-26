import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

def input_approximation(method, X, T, W_est, C_true=None, epsilon=0.1):
    '''
    Approximating the root causes C based 
    C ~ X - XA, where A is the block toeplitz matrix.
    '''
    d = W_est.shape[0]
    p = W_est.shape[1] // d - 1

    flag=False
    if len(X.shape) > 2: 
        flag=True
        n = X.shape[0]
        X = X.reshape((n, d * T))

    if X.shape[1] != d * T:
        a, _ = X.shape
        X = X[:int(a / T) * T, :]
        X = X.reshape((int(a / T), d * T))

    # block_matrix = utils.block_toeplitz(W_est, T=t)
    # dT = block_matrix.shape[0]

    W_est_list = [W_est[:, i * d : (i + 1) * d] for i in range(p + 1)]
    B = np.concatenate(W_est_list[::-1], axis=0) # B = [B_p
                                                #      B_{p-1}
                                                #      ...
                                                #      B_2
                                                #      B_1
                                                #      B_0  ]
    # # estimating the root causes
    # inverse_refl_trans_clos = np.eye(dT) - block_matrix if trans_clos == 'FW' else np.linalg.inv(np.eye(dT) + W_est)
    # C_est = X @ inverse_refl_trans_clos
    C_est = np.zeros(X.shape)
    for t in range(T - 1):
        if t < p:
            Y = X[:, 0 : (t + 1) * d] @ B[- ((t + 1) * d):, :]  # y = [x(0) ... x(t-1) x(t)] B 
        else:
            Y = X[:, (t - p) * d : (t + 1) * d] @ B # y = [x(t-p) ... x(t)] B 

        C_est[:, t * d : (t + 1) * d] = X[:, t * d : (t + 1) * d] - Y # c[t] = x[t] - x[t]A - x[t-1]B_1 -...-x[t-p]B_p

    if flag: 
        C_est = C_est.reshape(n, T, d)

    # if method not in ['spinsvar', 'sparserc', 'varlingam', 'd_varlingam'] only top-performing algorithms
    if C_true is None:
        return C_est, float("nan"), float("nan"), float("nan") # in real datasets the root cause ground truth is unknown
    
    else:
        # evaluating the estimation
        c_nmse = np.linalg.norm(C_est - C_true) / np.linalg.norm(C_true)

        ones_est = np.where(C_est > epsilon, 1, 0)
        ones_true = np.where(C_true > 0, 1, 0)
        c_shd = (ones_est * (1 - ones_true) + (1 - ones_est) * (ones_true)).sum() # total disagreement
        c_total = ones_true.sum() # total root causes
        
        return C_est, c_nmse, c_total, c_shd



def visualize_input(C_true, non_zero_true, C_est, non_zero_est, method):
    n, d = C_est.shape
    
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
        gray = cm.get_cmap('gray', 4)
        newcolors = gray(np.linspace(0, 1, 4))
        white = np.array([1, 1, 1, 1])
        black = np.array([0, 0, 0, 1])
        red = np.array([1, 0, 0, 1])
        grey = np.array([0.5, 0.5, 0.5, 1])
        newcolors[0, :] = white
        newcolors[1, :] = grey
        newcolors[2, :] = red
        newcolors[3, :] = black
        custom_cmp = ListedColormap(newcolors)

        l2 = np.where(non_zero_est != 0, 1, 0)

        common_l2 = non_zero_true * l2
        wrong_l2 = l2 - common_l2
        missed_l2 = non_zero_true - common_l2
        l2 = common_l2 + 0.66 * wrong_l2 + 0.33 * missed_l2

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(non_zero_true, cmap=custom_cmp)
        ax1.grid(False)
        ax1.add_patch(Rectangle((-0.5,-0.5), n - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax1.axis('off')
        ax1.set_title('Ground Truth')

        ax2.imshow(l2, cmap=custom_cmp)
        ax2.grid(False)
        ax2.add_patch(Rectangle((-0.5,-0.5), n - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax2.axis('off')
        ax2.set_title('Estimated')

        fig.suptitle('Root causes')

        plt.savefig('plots/root_causes_{}.pdf'.format(method), dpi=1000)
