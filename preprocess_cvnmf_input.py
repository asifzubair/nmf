#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:45:00 2021

@author: asif
"""

#import os
import sys
import pandas as pd
#import scipy.sparse as sp
import scanpy as sc

sys.path.append("../cNMF")
from cnmf import save_df_to_npz, load_df_from_npz
import nmf_helpers as nmh


counts_fn = "./data/GOSH/cnmf_input_GOSH.tsv"
counts = pd.read_table(counts_fn, index_col=0)
save_df_to_npz(counts, './data/GOSH/cnmf_input_GOSH.npz')

counts = load_df_from_npz('./data/GOSH/cnmf_input_GOSH.npz')
input_counts = sc.AnnData(X=counts.values,
                          obs=pd.DataFrame(index=counts.index),
                          var=pd.DataFrame(index=counts.columns)) 
tpm = nmh.compute_tpm(input_counts)

highvargenesfn = "./data/GOSH/cNMF_GOSH.overdispersed_genes.txt"
highvargenes = open(highvargenesfn).read().rstrip().split('\n')

norm_counts = nmh.get_norm_counts(input_counts, tpm, 
                                  num_highvar_genes=None,
                                  high_variance_genes_filter=highvargenes)

from cv_nmf import my_plot_nmf, my_plot_pca

my_plot_nmf(norm_counts.X, truth=15, replicates=5, p_holdout=0.1, 
            k0=2, k=30, 
            fig_name="./nmf_cv_curve_gosh_data_r_5_pholdout_0.1_k_2_29_run_1.pdf")

my_plot_pca(norm_counts.X, truth=15, replicates=5, p_holdout=0.1, 
            k0=2, k=30, 
            fig_name="./nmf_cv_curve_gosh_data_pca_r_5_pholdout_0.1_k_2_29_run_1.pdf")