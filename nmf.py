#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:11:53 2021

@author: asif zubair
"""

import pandas as pd
#import numpy as np
#import scipy.sparse as sp

#from sklearn.utils import sparsefuncs
#from sklearn.metrics import mean_squared_error
#from sklearn.decomposition import non_negative_factorization

import scanpy as sc
import nmf_helpers as nmh

input_counts  = pd.read_csv("./simulated_example_data/filtered_counts.txt", sep = "\t", index_col=0)

"""
if sp.issparse(tpm.X):
    gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
    gene_tpm_stddev = nmh.var_sparse_matrix(tpm.X)**.5
else:
    gene_tpm_mean = np.array(tpm.X.mean(axis=0)).reshape(-1)
    gene_tpm_stddev = np.array(tpm.X.std(axis=0, ddof=0)).reshape(-1)
 
# this is required for consensus NMF    
input_tpm_stats = pd.DataFrame([gene_tpm_mean, gene_tpm_stddev],
                               index = ['__mean', '__std'])
"""

umis = input_counts
#if umis.dtype != np.int64:
#    umis = umis.astype(np.int64)
umis_X, Y = nmh.split_molecules(umis, 0.9)

# conv_exp = nmh.convert_expectations(pca_X, data_split, data_split_complement)
# mcv_loss[i, j] = mean_squared_error(umis_Y, conv_exp)

input_counts = sc.AnnData(X=input_counts.values,
                          obs=pd.DataFrame(index=input_counts.index),
                          var=pd.DataFrame(index=input_counts.columns)) 
tpm = nmh.compute_tpm(input_counts)
norm_counts = nmh.get_norm_counts(input_counts, tpm, num_highvar_genes = 1500)

from cv_nmf import my_plot_nmf
my_plot_nmf(norm_counts.X, 5, 8)
my_plot_nmf(norm_counts.X, 5, 8, fig_name="./nmf_cv_curve_r_5_k_7_run_1.pdf")
my_plot_nmf(norm_counts.X, 5, 8, fig_name="./nmf_cv_curve_r_5_k_7_run_2.pdf")
my_plot_nmf(norm_counts.X, 5, 8, fig_name="./nmf_cv_curve_r_5_k_7_run_3.pdf")
my_plot_nmf(norm_counts.X, 5, 9, fig_name="./nmf_cv_curve_r_5_k_8_run_2.pdf")
my_plot_nmf(norm_counts.X, 5, 9, fig_name="./nmf_cv_curve_r_5_k_8_run_3.pdf")
my_plot_nmf(norm_counts.X, 5, 10, fig_name="./nmf_cv_curve_r_5_k_9_run_1.pdf")
my_plot_nmf(norm_counts.X, 5, 10, fig_name="./nmf_cv_curve_r_5_k_9_run_2.pdf")
my_plot_nmf(norm_counts.X, 5, 10, fig_name="./nmf_cv_curve_r_5_k_9_run_3.pdf")

my_plot_nmf(norm_counts.X, 5, k0=1, k=8, fig_name="./nmf_cv_curve_r_5_k_1_7_run_1.pdf")

my_plot_nmf(norm_counts.X, 5, k0=5, k=9, fig_name="./nmf_cv_curve_r_5_k_8_run_1.pdf")

my_plot_nmf(norm_counts.X, 5, k0=5, k=10, fig_name="./nmf_cv_curve_r_5_k_5_9_run_1.pdf")
my_plot_nmf(norm_counts.X, 5, k0=5, k=10, fig_name="./nmf_cv_curve_r_5_k_5_9_run_2.pdf")

my_plot_nmf(norm_counts.X, 10, k0=1, k=15, fig_name="./nmf_cv_curve_r_5_k_1_14_run_1.pdf")
