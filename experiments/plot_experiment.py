import sys
import os
import pandas as pd

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from experiments import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from matplotlib.colors import LinearSegmentedColormap

# adding gillsans font
from matplotlib import font_manager
font_dirs = ['experiments/plots_UAI/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# Define custom colormap
sunset_colors = [
    "#364B9A", "#4A7BB7", "#6EA6CD", "#98C6E2", "#C2E4EF",
    "#EAECCC", "#FEDA8B", "#FDB366", "#F67E4B"
]
sunset_cmap = LinearSegmentedColormap.from_list("tol_sunset", sunset_colors)


def histogram(C, dataset="sachs"):
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        plt.figure()
        plt.hist(C.flatten(), 100, color='blue')
        plt.xlabel('$|c_{ij}|$', fontsize=28, color='black')
        plt.ylabel('Count of values', fontsize=28, color='black')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(axis='y', color='white')
        plt.grid(axis='x', color='#e5e5e5')
        # plt.legend(frameon=False, fontsize=18) 
        plt.tight_layout()
        plt.savefig('plots_UAI/plot_{}_root_values_histogram.pdf'.format(dataset))

        plt.figure()
        plt.imshow(C.T, extent = [0, 852, 0, 10], aspect = 852/40, cmap='Blues')
        plt.grid(axis='y', color='#e5e5e5')
        plt.xlabel('Rows (samples)', color='black')
        plt.ylabel('Columns (nodes)', color='black')
        plt.tick_params( bottom=False, labelleft=False, labelbottom=False)
        plt.yticks(range(11))
        # plt.legend(frameon=False, fontsize=18) 
        # plt.tight_layout()
        plt.savefig('plots_UAI/plot_{}_root_spikes.pdf'.format(dataset), bbox_inches='tight')


def visualize(ground_truth, estimated, method='sparserc', filename='', args=None):
    d = ground_truth.shape[0]
    
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

        estimated = np.where(estimated != 0, 1, 0)

        common_estimated = ground_truth * estimated
        wrong_estimated = estimated - common_estimated
        missed_estimated = ground_truth - common_estimated
        estimated = common_estimated + 0.66 * wrong_estimated + 0.33 * missed_estimated

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(ground_truth, cmap=custom_cmp)
        ax1.grid(False)
        ax1.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax1.axis('off')
        ax1.set_title('Ground Truth')

        ax2.imshow(estimated, cmap=custom_cmp)
        ax2.grid(False)
        ax2.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax2.axis('off')
        ax2.set_title('{}'.format(method))

        fig.suptitle('Adjacency matrices')

        if filename=='':
            plt.savefig('experiments/plots_UAI/matrix_comparison/matrix_comparison_sachs.pdf')
        else:
            plt.savefig('experiments/plots_UAI/matrix_comparison/matrix_comparison_{}_{}.pdf'.format(filename, method), dpi=1000)

def visualize_stocks_adjacency(W_est, p, t, args, method='sparserc', ind=None, run=0):
    d = W_est.shape[0]
    if ind is None:
        ind = np.arange(d)
    
    df = pd.read_csv("experiments/data/S&P500/stock_weights.csv")
    df = df.loc[ind]
    df = df[['Symbol', 'categorical_index', 'alphabetical_index']].sort_values(by='categorical_index')
    ticks_index = df['categorical_index'].values
    data_index = df['alphabetical_index'].values # the alphabetical index is the order the algorithm has learned from the data

    df = pd.read_csv("experiments/data/S&P500/stock_weights.csv")
    df = df[['Symbol', 'GICS Sector', 'sort']].sort_values(by=['sort', 'Symbol'])
    ticks = df['Symbol'].to_numpy()[ticks_index]
    categories = df["GICS Sector"].to_numpy()[ticks_index]
    ps = [-1]
    labels = [categories[0]]
    for i in range(len(ind) - 1):
        if (categories[i] != categories[i+1]):
            labels.append(categories[i+1])
            ps.append(i)
    ps.append(len(ind) - 1)

    for i in range(p + 1):
        with plt.style.context('ggplot'):
            plt.rcParams['font.family'] = 'gillsans'
            plt.rcParams['xtick.color'] = 'black'
            plt.rcParams['ytick.color'] = 'black'

            fig, ax = plt.subplots(dpi=300, figsize = (9, 9))
            plt.imshow(W_est[:d, i * d : (i + 1) * d][data_index].T[data_index].T, cmap="seismic", vmax=0.5, vmin=-0.5)
            plt.grid(False)
            # ax.add_patch(Rectangle((-0.5,-0.5), len(ind) - 0.15, len(ind) - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
            for k in range(len(ps) - 1):
                ax.add_patch(Rectangle((ps[k] + 0.5, ps[k] + 0.5), ps[k + 1] - ps[k], ps[k + 1] - ps[k], linewidth=1, edgecolor='black', facecolor='none', label=labels[k]))
            # for k in range(len(ps) - 1):
            #     if k < (len(ps) - 1) / 2:
            #         plt.text(ps[k + 1] + 1, (ps[k] + 1 + ps[k + 1]) / 2 + 0.5,  labels[k], fontsize=9)
            #     if k >= (len(ps) - 1) / 2:
            #         text_size = plt.gcf().dpi * 0.0011 * len(labels[k])
            #         plt.text(ps[k] - text_size, (ps[k] + 1 + ps[k + 1]) / 2 + 0.5,  labels[k], fontsize=9)
                    
            # plt.axis('off')
            # plt.title('Estimated adjacency, lag = {}'.format(i))
            plt.xticks(ticks=range(len(ind)), labels=ticks, fontsize=9, rotation=90)
            ax.xaxis.tick_top()
            plt.yticks(ticks=range(len(ind)), labels=ticks, fontsize=9)
            ax.grid(color='gray', linestyle='--', linewidth=0.1)
            cbar = plt.colorbar(ax=ax, shrink=0.8)
            # Set the size of the colorbar ticks
            cbar.ax.tick_params(labelsize=14)  # Set the font size to 14


            plt.savefig('experiments/plots_UAI/S&P500/matrix_S&P_{}_lag_{}_timesteps_{}_l1_{}_l2_{}_omega_{}_run_{}.pdf'.format(method, i, t, args.lambda1, args.lambda2, args.omega, run), bbox_inches="tight")
            plt.close()


def visualize_root_causes(C_est, T, args, method="spinsvar", indx=None, indy=None, run=0):
    samples = C_est.shape[0]
    dT = C_est.shape[1]
    assert dT % T == 0
    d = dT // T
    C_est = C_est.reshape(samples * T, dT // T)

    if indx is None:
        indx = np.arange(d)
    if indy is None:
        indy = np.arange(samples * T)
    
    df = pd.read_csv("experiments/data/S&P500/stock_weights.csv")
    df = df.loc[indy]
    df = df[['Symbol', 'categorical_index', 'alphabetical_index']].sort_values(by=['categorical_index'])
    ticks_index = df['categorical_index'].values
    data_index = df['alphabetical_index'].values


    df = pd.read_csv("experiments/data/S&P500/stock_weights.csv")
    df = df[['Symbol', 'GICS Sector', 'sort']].sort_values(by=['sort', 'Symbol'])
    xticks = df['Symbol'].to_numpy()[ticks_index]

    dates = pd.read_csv("experiments/data/S&P500/Dates.csv")
    yticks = dates['Date'].to_numpy()[indx]

    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        fig, ax = plt.subplots(dpi=300, figsize = (12, 9))
        plt.imshow((C_est[indx].T[data_index]), cmap="seismic", vmax=0.25, vmin=-0.25, aspect=0.75) #"seismic"
        plt.grid(False)
        # ax.add_patch(Rectangle((-0.5,-0.5), len(indx), len(indy), linewidth=2, edgecolor='black', facecolor='none'))
                
        # plt.title('Estimated root causes')
        plt.yticks(ticks=range(len(xticks)), labels=xticks, fontsize=9)
        plt.xticks(ticks=range(len(yticks)), labels=yticks, fontsize=11, rotation=90)
        ax.xaxis.tick_top()
        ax.grid(color='gray', linestyle='--', linewidth=0.1)
        cbar = plt.colorbar(ax=ax, shrink=0.8)

        # Set the size of the colorbar ticks
        cbar.ax.tick_params(labelsize=14)  # Set the font size to 14

        plt.savefig('experiments/plots_UAI/S&P500/RootCauses_{}_timesteps_{}_l1_{}_l2_{}_omega_{}_run_{}.pdf'.format(method, T, args.lambda1, args.lambda2, args.omega, run), bbox_inches="tight")
        plt.close()

def plot_accuracy(avg, std, x_axis, methods, param='nodes', filename='default', legend=False):
    full = 'MMPC' in methods # full version of plot with all methods

    linewidth = {}
    markersize = {}
    for method in color_methods.keys():
        linewidth[method] = 1.5
        markersize[method] = 6
    markersize['spinsvar'] = 10
    linewidth['spinsvar'] = 5
    # linewidth['pc'] = 3
    # markersize['sparserc'] = 6
    # linewidth['sparserc'] = 2

    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.frameon'] = True
        
        for i, label in enumerate(['nSHD', 'SHD', 'TPR', 'NNZ', 'PREC', 'REC', 'F$1$-score', 'AUROC', 'NMSE', 'Time [s]', 'SID', '$\widehat{\mathbf{\mathcal{S}}}$ NMSE', '$\widehat{\mathbf{\mathcal{S}}}$ NUM', '$\widehat{\mathbf{\mathcal{S}}}$ SHD']): #
            if param != 'weight_bounds':
                if (not legend):
                    fig, ax = plt.subplots()
                    ax.spines['bottom'].set_color('black')
                    ax.spines['left'].set_color('black') 

                    plt.xscale('log')
                    plt.minorticks_off()

                    for method in methods:
                        if(len(avg[method] > 0)):
                            # ticks = np.arange(len(avg[method][:, i]))
                            # print(avg[method].shape)
                            plt.plot(x_axis[:len(avg[method][:, i])], avg[method][:, i], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method], linestyle = 'solid', marker=marker_methods[method], markersize=markersize[method])
                            plt.fill_between(x_axis[:len(avg[method][:, i])], avg[method][:, i] - std[method][:, i], avg[method][:, i] + std[method][:, i], color=color_methods[method], alpha=.1)

                    f = False
                    if filename == 'samples_[1]_timeout_10000_':
                        plt.ylabel(label, fontsize=30, color='black')
                        lims["SHD"]=[-50, 1050]
                        f = True
                    # plt.xlabel('Number of sample sizes' if args.value=='samples' else 'Number of vertices', fontsize=20)
                    # if (args.samples == [1] or args.weight_bounds[1] == 0.2): #if (label == 'SID' and not full) or (full and label == 'NMSE'):
                    
                    # if f:
                    plt.ylabel(label, fontsize=35, color='black')

                    # if label in ['Time [s]', '$\mathbf{C}$ NMSE']:
                    plt.xlabel('Nodes $d$' if param=='nodes' else 'Samples $N$' if param=='samples' else 'Timesteps' if param == 'timesteps' else 'Sparsity %' if param=='sparsity' else 'Noise std.' if param=='noise_std' else 'Input lag $k\'$' if param=='algo_lags' else 'Avg. degree', fontsize=35, color='black')
                    # ticks = np.arange(len(x_axis))
                    if args.rotate == "True":
                        plt.xticks(x_axis, x_axis, fontsize=22, rotation=45)
                    else:
                        plt.xticks(x_axis, x_axis, fontsize=22)

                    # Get the current yticks
                    # current_yticks = ax.get_yticks()
                    # print(current_yticks)
                    # Set every second ytick
                    # sparse_yticks = current_yticks[1::2]
                    # print(sparse_yticks)
                    # ax.set_yticks(sparse_yticks)
                    # ax.tick_params(axis='y', labelsize=22)
                    plt.locator_params(axis='y', nbins=5) 
                    plt.yticks(fontsize=22)

                    # plt.grid(axis='y', color='white')
                    # plt.grid(axis='x', color='white')#e5e5e5
                    # if(f):
                    #     plt.legend(frameon=False, fontsize=24, loc='upper left') 
                    # if label == 'SHD' and param != 'nodes':
                    #     plt.ylim([0, 2 * 100])
                    if label in lims.keys():
                        plt.ylim(lims[label])
                    plt.tight_layout()

                    fullname = '_full' if full else ''
                    plt.savefig('experiments/plots_UAI/plot{}_{}_{}.pdf'.format(fullname, filename, file_label[label]), bbox_inches="tight")

                # # only print legend
                elif (i == 0):
                    plt.figure()

                    plt.rcParams['axes.facecolor']='white'
                    plt.rcParams['savefig.facecolor']='white'
                    for method in methods:
                        if(len(avg[method] > 0)):
                            plt.plot([], [], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method], marker=marker_methods[method], markersize=markersize[method], linestyle = 'dashed' if method=='pc' else 'solid')
                    plt.xticks([])
                    plt.yticks([])
                    plt.legend(frameon=False, fontsize=20)
                    plt.tight_layout()
                    fullname = '_full' if full else ''
                    plt.savefig('experiments/plots_UAI/plot{}_{}_legend_only.pdf'.format(fullname, filename), bbox_inches='tight')
            
            else: 
                bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    
                y, x = np.meshgrid(bounds, bounds)
                
                z = {}
                for method in methods:
                    z[method] = np.zeros(x.shape)
                
                for j in range(len(x_axis)):
                    v = x_axis[j]
                    v0 = int(10 * v[0]) - 1
                    v1 = int(10 * v[1]) - 1
                    for method in methods:
                        z[method][v0, v1] = avg[method][j, i]
                    
                for method in methods:
                    if(len(avg[method] > 0)):
                        plt.figure()
                        z_min, z_max = np.abs(z[method]).min(), np.abs(z[method]).max()
                        plt.pcolormesh(x, y, z[method], cmap='RdBu', vmin=z_min, vmax=z_max)# avg[method][:, i], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method])
                        plt.colorbar()
                        plt.xlabel('Lower bound a')
                        plt.ylabel('Upper bound b')
                        for i in range(z[method].shape[0]):
                            for j in range(z[method].shape[1]):
                                plt.text(j, i, '{:.2f}'.format(z[method][i, j]), ha='center', va='center')
                        plt.savefig('plots_UAI/plot_{}_{}_{}.pdf'.format(filename, method, label))
                

def append_info(avg, std, info, method):
    if(info[0] == 'Acc {} is'.format(method)):
        avg[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8]), float(info[9]), float(info[10]), float(info[11]), float(info[12]), float(info[13]), float(info[14])])
    elif(info[0] == 'Std {} is'.format(method)):
        std[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[6]), float(info[7]), float(info[8]), float(info[9]), float(info[10]), float(info[11]), float(info[12]), float(info[13]), float(info[14])])


def plot_accuracy_vs_param(args, param='sparsity'):
    dic = vars(args)

    avg = {}
    std = {}
    for key in methods:
        avg[key] = []
        std[key] = []

    if param == 'nodes':
        values = args.nodes
    elif param == 'samples':
        values = args.samples
    elif param == 'timesteps':
        values = args.timesteps
    elif param == 'sparsity':
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    elif param == 'weight_bounds':
        bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    
        values = [[a, b] for a in bounds for b in bounds if a <= b]
    elif param == 'noise_std':
        values = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
    elif param == 'edges':
        values = [5, 10, 15, 20]
    elif param == 'algo_lags':
        values = [1, 2, 3, 4, 5, 6]
    else:
        print("case not covered")

    if param in ['nodes', 'samples', 'timesteps']:
        # finding the name of the output files according to the experimental settings 
        filename, _ = utils.get_filename(parser, args)

        with open('results_UAI/AVG_{}.csv'.format(filename), 'r') as f:
            for line in f:
                info = line.split(',')
                for method in methods:
                    append_info(avg, std, info, method)

    elif param in ['algo_lags']:
        for v in values:
            filename, _ = utils.get_filename(parser, args)
            if v == parser.get_default(param):
                filename = filename
            else:
                filename += '{}_{}_'.format(param, v)

            with open('results_UAI/AVG_{}.csv'.format(filename), 'r') as f:
                    for line in f:
                        info = line.split(',')
                        for method in methods:
                            append_info(avg, std, info, method)
    else:
        for v in values:
            if v == parser.get_default(param):
                filename = 'default'
            else:
                if param in ['weight_bounds', 'edges']:
                    filename = '{}_{}_'.format(param, v)
                else:
                    filename = '{}_{:.1f}_'.format(param, v)

            # nodes_flag = False # flag will only be true when we see the results about the 30 nodes
            with open('results_UAI/AVG_{}.csv'.format(filename), 'r') as f:
                for line in f:
                    info = line.split(',')
                    # if len(info) > 9:
                    #     if (info[9] == ' nodes = [30]'):
                    #         nodes_flag = True 
                    #     else:
                    #         nodes_flag = False
                    # elif nodes_flag:
                    for method in methods:
                        append_info(avg, std, info, method)
                        
        filename = param

    for method in methods:
        avg[method] = np.array(avg[method])
        std[method] = np.array(std[method])

    x_axis = values
    plot_accuracy(avg, std, x_axis, methods, param=param, filename=filename, legend=(args.legend == 'True'))


if __name__ == '__main__':
    parser, args = utils.get_args()

    samples = args.samples #[200, 400, 600, 800, 1000] if(args.value == 'samples') else [400]
    variables =  args.nodes # if(args.value == 'variables') else [10] 
    noise = args.noise
    runs = args.runs 
    (a, b) = tuple(args.weight_bounds)
    k = args.edges
    methods = args.methods

    color_methods = {
        'spinsvar': '#A50026', 
        'nts-notears' : '#DD3D2D',
        'd_varlingam' : '#F67E4B',
        'culingam' : '#FDB366',
        'sparserc' : '#FEDA8B',
        'dynotears' : '#EAECCC', 
        'TCDF' : '#98CAE1', 
        'pcmci' : '#6EA6CD',
        'tsfci' : '#4A7BB7',
        'varlingam': '#364B9A', 
    }

    label_methods = {
        'spinsvar': 'SpinSVAR (Ours)', 
        'sparserc': 'SparseRC',
        'varlingam' : 'VARLiNGAM',
        'd_varlingam' : 'Directed VARLiNGAM',
        'culingam' : 'cuLiNGAM',
        'dynotears': 'DYNOTEARS', 
        'nts-notears' : 'NTS-NOTEARS', 
        'tsfci' : 'tsFCI',
        'pcmci' : 'PCMCI',
        'TCDF': 'TCDF', 
        'lingam' : 'LiNGAM',
        'GES': 'GES', 
        'MMPC': 'MMHC', 
        'CAM' : 'CAM', 
        'FGS' : 'fGES',
        'sortnregress': 'sortnregress',
        'pc' : 'PC'
    }

    marker_methods = {
        'spinsvar': '*',       # star
        'sparserc': 'd',       # diamond
        'varlingam': 'p',      # plus (filled)
        'd_varlingam': 'o',    # circle
        'culingam': '^',       # triangle up
        'dynotears': 'v',      # triangle down
        'nts-notears': '<',    # triangle left
        'tsfci': '>',          # triangle right
        'pcmci': 's',          # square
        'TCDF': 'X',           # X-shaped marker
    }


    file_label = {
        'nSHD': 'nshd',
        'SHD' : 'shd',
        'TPR' : 'tpr',
        'NNZ' : 'nnz', 
        'FPR' : 'fpr', 
        'SID' : 'sid', 
        'NMSE': 'nmse', 
        '$\widehat{\mathbf{\mathcal{S}}}$ NUM' : 'c_num', 
        'Time [s]' : 'time', 
        '$\widehat{\mathbf{\mathcal{S}}}$ NMSE' : 'c_nmse',
        '$\widehat{\mathbf{\mathcal{S}}}$ SHD' : 'c_shd',
        'ACC' : 'acc',
        'PREC' : 'prec',
        'REC' : 'rec',
        'F$1$-score' : 'F1',
        'AUROC' : 'AUROC'
    }

    lims = {
        'nSHD': [0, 0.5],
        'SHD' : [0, 2000], 
        # 'Time (s)' :  [0, 250],
        '$\widehat{\mathbf{\mathcal{S}}}$ TPR' : [0.75, 1],
        '$\widehat{\mathbf{\mathcal{S}}}$ SHD' : [0, 60000]
    }

    if len(args.nodes) > 1:
        plot_accuracy_vs_param(args, param='nodes')
    elif len(args.samples) > 1:
        plot_accuracy_vs_param(args, param='samples')
    elif len(args.timesteps) > 1: 
        plot_accuracy_vs_param(args, param='timesteps')

    # plot_accuracy_vs_param(args, param='sparsity')

    # plot_accuracy_vs_param(args, param='noise_std')

    # plot_accuracy_vs_param(args, param='weight_bounds')

    # plot_accuracy_vs_param(args, param='edges')

    # plot_accuracy_vs_param(args, param='algo_lags')
