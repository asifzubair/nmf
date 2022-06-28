#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:47:10 2021

@author: asif
"""

import os, sys
import pandas as pd
#import scipy.sparse as sp
import scanpy as sc

sys.path.append("../cNMF")
from cnmf import load_df_from_npz
import nmf_helpers as nmh

base_directory = './data/Part3_VisualCortex/'
countfn = os.path.join(base_directory, 'GSE71585_RefSeq_TPM.filtered.npz')
K = ' '.join([str(k) for k in range(15,27)])

highvargenesfn = os.path.join(base_directory,'GSE71585_RefSeq_filt_TPM_2000hvgs.txt')
tpmfn = countfn

input_counts = load_df_from_npz(countfn)
input_counts = sc.AnnData(X=input_counts.values,
                          obs=pd.DataFrame(index=input_counts.index),
                          var=pd.DataFrame(index=input_counts.columns))

tpm = load_df_from_npz(tpmfn)
tpm = sc.AnnData(X=tpm.values,
                 obs=pd.DataFrame(index=tpm.index),
                 var=pd.DataFrame(index=tpm.columns))

highvargenes = open(highvargenesfn).read().rstrip().split('\n')
norm_counts = nmh.get_norm_counts(input_counts, tpm, 
                                  num_highvar_genes=None,
                                  high_variance_genes_filter=highvargenes)

from cv_nmf import my_plot_nmf
my_plot_nmf(norm_counts.X, replicates=5, k0=15, k=27, fig_name="./nmf_cv_curve_tasic_data_r_5_k_15_26_run_1.pdf")

#my_plot_nmf(norm_counts.X, 
#            truth=20, 
#            replicates=5, 
#            k0=5, k=20, 
#            fig_name="./nmf_cv_curve_tasic_data_r_5_k_5_20_run_1.pdf")