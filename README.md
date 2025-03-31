# Code reposity for Danielle Paulson's Harvard College Senior Thesis "Leveraging Sparsity for Multi-Parameter Inference in the Gaussian Linear Model"

This repository contains the source code for implementing all of the testing procedures and reproducing all of the figures presented in the thesis. The L-test, a multivariate extension of the work in https://arxiv.org/abs/2406.18390, is a LASSO-based method for testing the significance of a subset of parameters of interest. It has higher power than the classic F-test when the true parameter vector is sparse while providing the same validity guarantees under the exact same assumptions. Because it uses a Monte Carlo p-value that requires resampling and running the group LASSO numerous times, it is computationally intensive to perform. For this reason, we also proposed a family of alternative tests called recentered F-tests that approximate the L-test while being significantly more computationally efficient. The source code includes implementations of both the L-test and recentered F-tests.

## Methods Implementation
The file `tests.py` contains the source code for implementing the L-test and its modifications when alternative LASSO-based estimates are used (ie. standard LASSO, elastic net, sparse group LASSO), the recentered F-tests, and the oracle-test. 

## Reproducing Figures
The folder `reproducibility` contains the code for reproducing all of the experiments. Some of these were run on a single computer while others were run on a computing cluster with a Slurm manager. For these instances, we provide the scripts needed for submitting numerous jobs to the cluster in `reproducibility/scripts`. Note that a user may need to update these scripts depending on the file locations, partition names, etc. Below, we specify how to use the files in the `reproducibility` folder to generate each of the figures shown.
1. Power plots: For Figures 4.1.1, 4.1.2, 5.5.1, and 5.5.2,
2. Heat maps: For Figure 4.1.3,
3. Efficiency tables: For Tables 4.2.1 and 4.2.2,
4. Robustness plots: For Figures 5.5.5, 5.5.6, 5.5.7, and 5.5.8 . The panels in Figure 4.3.1 are taken from those in the aforementioned figures.
5. HIV analysis: For Figures 4.4.1 and 5.5.9, 
6. Penalty tuning:
7. Visualizing fcn g: For Figures 5.5.3 and 5.5.4, 
