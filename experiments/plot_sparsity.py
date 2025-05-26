import matplotlib.pyplot as plt
import numpy as np

def plot_sparsity(bernoulli, gauss, laplace, threshold):
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.frameon'] = True
            
        fig, ax = plt.subplots()
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black') 

        # plt.xscale('log')
        plt.minorticks_off()

        bins = [0, 0.5 * threshold, threshold, 1.5 * threshold, 3 * threshold, 4.5 * threshold]
        bin_labels = ["{:.2f}".format(s) for s in bins]
        plt.hist([np.abs(bernoulli), np.abs(gauss), np.abs(laplace)], bins=bins)
        plt.legend(labels=["Bernoulli", "Gauss", "Laplace"])
        plt.xticks(bins, bin_labels, fontsize=12)
        plt.show()
        plt.savefig('experiments/plots/sparsity_population.pdf', bbox_inches="tight")