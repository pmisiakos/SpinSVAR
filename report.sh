# # ################# Laplace #############################################
# ################# samples 1, length 1000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000  --runs 5 --sparsity_type laplace --timeout 10000 --rotate True
# python experiments/plot_experiment.py --methods spinsvar --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0  --noise_std 1  --rotate True
# python experiments/plot_experiment.py --methods spinsvar --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0  --noise_std 1  --legend True

# Laplace N=1, T=10000
python experiments/plot_experiment.py --methods spinsvar varlingam d_varlingam culingam sparserc dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 20 30 50 100 200 500 1000  --runs 5 --sparsity_type laplace --timeout 10000 --rotate True

# ################# samples 10, length 1000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF  --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000 --rotate True
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF  --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity_type laplace --timeout 10000 --rotate True
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF  --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity_type laplace --timeout 10000 --legend True

# ################## varying samples
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 2 3 5 10 20 --timesteps 1000 --nodes 500  --runs 5 --sparsity_type laplace --timeout 10000

# # ################# larger lag 
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 --runs 5 --sparsity_type laplace --timeout 10000

# # ############### lag sensitivity # first uncomment accuracy vs param
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam  --number_of_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity_type laplace --timeout 10000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam  --number_of_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity_type laplace --timeout 10000  --legend True


# # ################# Bernoulli + Uniform ##################################
# ################# samples 1, length 1000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000  --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000 --rotate True
# python experiments/plot_experiment.py --methods spinsvar --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0  --noise_std 1  --rotate True
# python experiments/plot_experiment.py --methods spinsvar --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0  --noise_std 1  --legend True

#  N=1, T=10000
python experiments/plot_experiment.py --methods spinsvar varlingam d_varlingam culingam sparserc dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 20 30 50 100 200 500 1000  --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000 --rotate True

# ################# samples 10, length 1000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF  --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000 --rotate True
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF  --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 4000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000 --legend True

# ################## varying samples
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 2 3 5 10 20 --timesteps 1000 --nodes 500  --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000

# # ################# larger lag 
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000 2000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000

# # ############### lag sensitivity # first uncomment accuracy vs param
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam  --number_of_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000
python experiments/plot_experiment.py --methods spinsvar sparserc varlingam  --number_of_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity 0.05 --sparsity_type uniform --timeout 10000  --legend True

# # ################# Bernoulli + Gauss ##################################
# N=10, T=1000
python experiments/plot_experiment.py --methods spinsvar varlingam d_varlingam culingam sparserc dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100 200 500 1000  --runs 5 --sparsity 0.05 --sparsity_type gauss --timeout 10000 --rotate True

# ################# Finance CPT
#best
python experiments/print_table_tex.py --dataset finance --methods spinsvar sparserc varlingam d_varlingam culingam TCDF
# appendix 
python experiments/print_table_tex.py --dataset finance --methods dynotears nts-notears tsfci pcmci

# # ################# Stocks 
python experiments/plot_stocks.py --methods  spinsvar sparserc varlingam  dynotears pcmci TCDF  --runs 1
