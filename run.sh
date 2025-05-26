################## Hyperparameter optimization synthetic data
# Laplace
python hyperparameter_search.py --methods spinsvar sparserc dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 --runs 5 --sparsity_type laplace 
# Bernoulli + Uniform
python hyperparameter_search.py --methods spinsvar sparserc dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 --runs 5 --sparsity 0.05

########### LAPLACE ##################################################################################################################################
################## samples 10, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100  --runs 5 --sparsity_type laplace   --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 200            --runs 5 --sparsity_type laplace   --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears TCDF             --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 500 1000       --runs 5 --sparsity_type laplace   --timeout 10000 #25hrs
python synthetic_experiment.py --methods spinsvar                                      --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 2000 4000      --runs 5 --sparsity_type laplace   --timeout 10000 #2.5hrs

################### samples 1, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100  --runs 5 --sparsity_type laplace   --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 200       --runs 5 --sparsity_type laplace   --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears                  --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 500       --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc                             --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 1000       --runs 5 --sparsity_type laplace   --timeout 10000

# N=1, T=10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 20 30 50 100  --runs 5 --sparsity_type laplace   --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 200       --runs 5 --sparsity_type laplace   --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears                  --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 500       --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc                             --number_of_lags 2 --samples 1 --timesteps 10000 --nodes 1000       --runs 5 --sparsity_type laplace   --timeout 10000

# ################## varying samples
python synthetic_experiment.py --methods spinsvar sparserc varlingam --number_of_lags 2 --samples 1 2 3 5 10 20 --timesteps 1000 --nodes 500  --runs 5 --sparsity_type laplace   --timeout 10000 # 5 x 6 x 20mins = 10hrs

############# Checking time-lag effect
################# samples 10, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 4 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 2 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 1 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam            --number_of_lags 3 --algo_lags 6 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity_type laplace   --timeout 10000 

################################### Large DAGs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 1000 --runs 1  --sparsity_type laplace   --timeout 10000 # 5mins
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 2000 --runs 1  --sparsity_type laplace   --timeout 10000 # 1hrs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 8 16 32 64 --timesteps 1000  --nodes 4000 --runs 1  --sparsity_type laplace   --timeout 10000 # 6hrs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 32         --timesteps 1000  --nodes 8000 --runs 1  --sparsity_type laplace   --timeout 10000 # timeout

python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 8 16             --timesteps 1000  --nodes 1000 --runs 1  --sparsity_type laplace   --timeout 10000 # 4hrs
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 1 2 4 8 16 32 64 --timesteps 1000  --nodes 2000 --runs 1  --sparsity_type laplace   --timeout 10000 # TIMEOUT everywhere
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 8 16 32 64       --timesteps 1000  --nodes 4000 --runs 1  --sparsity_type laplace   --timeout 10000 # TIMEOUT even at 8 so no chance
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 128              --timesteps 1000  --nodes 8000 --runs 1  --sparsity_type laplace   --timeout 10000 # Timeout

python synthetic_experiment.py --methods sparserc   --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 1000 --runs 1  --sparsity_type laplace   --timeout 10000 # 5mins
python synthetic_experiment.py --methods sparserc   --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 2000 --runs 1  --sparsity_type laplace   --timeout 10000 # 1hrs

#################################### More lags
python synthetic_experiment.py --methods spinsvar varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 20 30 50 --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam dynotears d_varlingam culingam TCDF --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 100 200 500 --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam dynotears                  --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 500 --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc                             --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity_type laplace   --timeout 10000
python synthetic_experiment.py --methods spinsvar                                      --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 2000 --runs 5 --sparsity_type laplace   --timeout 10000

###################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################
########### Bernoulli + Uniform ##################################################################################################################################
################### samples 1, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 20 30 50 100  --runs 5 --sparsity 0.05 --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 200       --runs 5 --sparsity 0.05 --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears                  --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 500       --runs 5 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc                             --number_of_lags 2 --samples 1 --timesteps 1000 --nodes 1000       --runs 5 --sparsity 0.05 --timeout 10000

################## samples 10, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100  --runs 5 --sparsity 0.05 --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 200            --runs 5 --sparsity 0.05 --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears TCDF             --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 500 1000       --runs 5 --sparsity 0.05 --timeout 10000 #25hrs
python synthetic_experiment.py --methods spinsvar                                      --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 2000 4000      --runs 5 --sparsity 0.05 --timeout 10000 #2.5hrs

# ################## varying samples
python synthetic_experiment.py --methods spinsvar sparserc varlingam --number_of_lags 2 --samples 1 2 3 5 10 20 --timesteps 1000 --nodes 500  --runs 5 --sparsity 0.05 --timeout 10000 # 5 x 6 x 20mins = 10hrs

############# Checking time-lag effect
################# samples 10, length 1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 3 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 4 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 2 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 1 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc varlingam   --number_of_lags 3 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam            --number_of_lags 3 --algo_lags 6 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000  --runs 2 --sparsity 0.05 --timeout 10000 

################################### Large DAGs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 1000 --runs 1  --sparsity 0.05 --timeout 10000 # 5mins
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 2000 --runs 1  --sparsity 0.05 --timeout 10000 # 1hrs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 8 16 32 64 --timesteps 1000  --nodes 4000 --runs 1  --sparsity 0.05 --timeout 10000 # 6hrs
python synthetic_experiment.py --methods spinsvar  --number_of_lags 2 --samples 32         --timesteps 1000  --nodes 8000 --runs 1  --sparsity 0.05 --timeout 10000 # timeout

python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 8 16             --timesteps 1000  --nodes 1000 --runs 1  --sparsity 0.05 --timeout 10000 # 4hrs
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 1 2 4 8 16 32 64 --timesteps 1000  --nodes 2000 --runs 1  --sparsity 0.05 --timeout 10000 # TIMEOUT everywhere
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 8 16 32 64       --timesteps 1000  --nodes 4000 --runs 1  --sparsity 0.05 --timeout 10000 # TIMEOUT even at 8 so no chance
python synthetic_experiment.py --methods varlingam  --number_of_lags 2 --samples 128              --timesteps 1000  --nodes 8000 --runs 1  --sparsity 0.05 --timeout 10000 # Timeout

python synthetic_experiment.py --methods sparserc   --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 1000 --runs 1  --sparsity 0.05 --timeout 10000 # 5mins
python synthetic_experiment.py --methods sparserc   --number_of_lags 2 --samples 1 2 4 8 16 --timesteps 1000  --nodes 2000 --runs 1  --sparsity 0.05 --timeout 10000 # 1hrs

#################################### More lags
python synthetic_experiment.py --methods spinsvar varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 20 30 50 --runs 5 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam dynotears d_varlingam culingam TCDF --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 100 200 500 --runs 5 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar varlingam dynotears                  --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 500 --runs 5 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar sparserc                             --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 1000 --runs 5 --sparsity 0.05 --timeout 10000
python synthetic_experiment.py --methods spinsvar                                      --number_of_lags 5 --algo_lags 5 --weight_bounds 0.1 0.2 --samples 10 --timesteps 1000 --nodes 2000 --runs 5 --sparsity 0.05 --timeout 10000

###################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################
########### Bernoulli + Gauss ##################################################################################################################################
# N=10, T=1000
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears nts-notears tsfci pcmci TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 20 30 50 100  --runs 5 --sparsity 0.05 --sparsity_type gauss --timeout 10000 # 5hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam d_varlingam culingam dynotears TCDF --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 200            --runs 5 --sparsity 0.05 --sparsity_type gauss --timeout 10000 #10hrs
python synthetic_experiment.py --methods spinsvar sparserc varlingam dynotears TCDF             --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 500 1000       --runs 5 --sparsity 0.05 --sparsity_type gauss --timeout 10000 #25hrs
python synthetic_experiment.py --methods spinsvar                                      --number_of_lags 2 --samples 10 --timesteps 1000 --nodes 2000 4000      --runs 5 --sparsity 0.05 --sparsity_type gauss --timeout 10000 #2.5hrs
