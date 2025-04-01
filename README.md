# Code reposity for Danielle Paulson's Harvard College Senior Thesis "Leveraging Sparsity for Multi-Parameter Inference in the Gaussian Linear Model"

This repository contains the source code for implementing all of the testing procedures and reproducing all of the figures presented in the thesis. The L-test, a multivariate extension of the work in https://arxiv.org/abs/2406.18390, is a LASSO-based method for testing the significance of a subset of parameters of interest in the Gaussian linear model with n > d. It has higher power than the classic F-test when the true parameter vector is sparse while providing the same validity guarantees under the exact same assumptions. Because it uses a Monte Carlo p-value that requires resampling and running the group LASSO numerous times, it is computationally intensive to perform. For this reason, we also proposed a family of alternative tests called recentered F-tests that approximate the L-test while being significantly more computationally efficient. The source code includes implementations of both the L-test and recentered F-tests.

## Methods Implementation
The file `tests.py` contains the source code for implementing the L-test and its modifications when alternative LASSO-based estimates are used (ie. standard LASSO, elastic net, sparse group LASSO), the recentered F-tests, and the oracle-test. 

## Reproducing Figures
The folder `reproducibility` contains the code for reproducing all of the experiments. Some of these were run on a single computer while others were run on a computing cluster with a Slurm manager. For these instances, we provide the scripts needed for submitting numerous jobs to the cluster in `reproducibility/scripts`. Note that a user may need to update these scripts depending on the file locations, partition names, etc. Below, we specify how to use the files in the `reproducibility` folder to generate each of the figures.
1. Power plots: Figures 4.1.1, 4.1.2, 5.5.1, and 5.5.2. Submit N=1000 jobs to the cluster with `sbatch --array=1-N run_tests.sh` to run the file `power_analysis.py` a total of N times, where N corresponds to the number of simulations. To combine the simulation results and compute the average powers, run the file `combine_sims.py` with the command `sbatch run_combine.sh`. This generates a `test_powers.pkl` file that stores the average powers of the tests. To generate the power plots, run the file `viz_powers.py` in the same directory where the `test_powers.pkl` file is stored. The `power_analysis.py` file has to be modified to generate each of the figures. For Figure 4.1.2, the `low-dimensional model setting` code block should be commented out, for Figure 5.5.1, the `high-dimensional model setting` code block should be commented out, and for Figure 5.5.2, the `large model setting` code block should be commented out. For Figure 4.1.1, the `low-dimensional model setting` code block should be commented out and the recentered F-tests that are called inside the `power` function should be swapped out for the standard LASSO test, elastic net test, and sparse group LASSO test implemented in the `tests.py` file.
2. Heat maps: Figure 4.1.3. Run the file `heatmaps.py.` To generate each row of panels in the figure, uncomment the piece of code noted in the file. 
3. Efficiency: Tables 4.2.1 and 4.2.2. Run the file `efficiency.py` and uncomment the code blocks corresponding to each table. To generate Table 4.2.2, in the `time_tests` function, also uncomment the lines pertaining to the L-test to get the L-test p-values and times.
4. Robustness plots: Figures 4.3.1, 5.5.5, 5.5.6, 5.5.7, and 5.5.8. Note that the panels in Figure 4.3.1 are taken from those in the other figures listed in the manner described in its caption. Submit N=1000 jobs to the cluster with `sbatch --array=1-N run_viols.sh` to run the file `violations.py` a total of N times, where N corresponds to the number of simulations. To combine the simulation results and compute the Type I error rates, run the file `combine_violations.py` with the command `sbatch run_combine_viols.sh`. This generates four groups of .pkl files, each corresponding to a different type of model violation and each containing two files, one corresponding to the L-test error rates and the other corresponding to the F-test error rates. To generate all of the panels that appear in Figures 5.5.5, 5.5.6, 5.5.7, and 5.5.8, run `viz_violations.py`, ensuring that it is in the same directory as the .pkl files. 
5. HIV analysis: Figures 4.4.1 and 5.5.9. Submit 16 jobs to the cluster with `sbatch --array=1-16 run_HIV.sh` to run the file `HIV.py` a total of 16 times, where each iteration corresponds to one of the 16 regressions. The p-values corresponding to the i-th regression get stored in `p_vals_reg_i.csv`. Move the `p_vals_reg_i.csv` files into a directory called `p_values` and then run `HIV_combine.py`, ensuring that this file is in the same directory as the `p_values` directory, to generate all of the plots that appear in the figures.
6. Penalty tuning: Figures 5.4.1, 5.4.2, and 5.4.3. To generate Figure 5.4.1, submit N=1000 jobs to the cluster with `sbatch --array=1-N run_rules.sh` to run the file `compare_rules.py` a total of N times, where N is the number of simulations. Run the file `combine_rules.py` with the command `sbatch run_combine_rules.sh`. This generates a `test_powers.pkl` file that stores the average powers of the tests. To make the plots, run the file `viz_rules.py` in the same directory where the `test_powers.pkl` file is stored. To obtain the plot corresponding to the high-dimensional setting, uncomment the `high-dimensional model setting` code block in the `compare_rules.py` file. To generate Figure 5.4.3, repeat this same process, except use the `run_CV_data.sh` script to run the `compare_CV_datasets.py` file, the `run_combine_CV_data.sh` script to run the `combine_CV_datasets.py` file, and the `viz_CV_datasets.py` file for creating the plots. To generate Figure 5.4.2, submit N=100 jobs to the cluster with `sbatch --array=1-N run_random.sh` to run the file `randomization.py` a total of N times, where N corresponds to the number of simulations. To combine the simulation results, run the file `combine_randomization.py` with the command `sbatch run_combine_random.sh`. This generates a total of six .pkl files, each containing the p-values corresponding to one of the six signal settings we considered. To make the plot, run the file `viz_randomization.py`, ensuring it is in the same directory as the .pkl files.
7. Visualizing fcn g: Figures 5.5.3 and 5.5.4. For the left panels of Figures 5.5.3 and 5.5.4, run files `viz_g_dim=2.py` and `viz_g_dim=3.py`, respectively. For the right panels corresponding to the high-dimensional setting, make the adjustments in the comments and run the same files.
