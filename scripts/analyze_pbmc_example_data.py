#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:06:38 2021

@author: asif
"""

# from scipy.io import mmread
# import scipy.sparse as sp
import matplotlib.pyplot as plt
import nmf_helpers as nmh

# import os
# import pandas as pd
import numpy as np
import scanpy as sc

np.random.seed(14)

adata = sc.read_10x_mtx(
    "example_PBMC/filtered_gene_bc_matrices/hg19/",
    var_names="gene_symbols",
    cache=False,
)
adata.var_names_make_unique()

sc.pp.filter_cells(adata, min_genes=200)  # filter cells with fewer than 200 genes
sc.pp.filter_cells(
    adata, min_counts=200
)  # This is a weaker threshold than above. It is just to population the n_counts column in adata
sc.pp.filter_genes(adata, min_cells=3)  # filter genes detected in fewer than 3 cells

## plot log10 # counts per cell
plt.hist(np.log10(adata.obs["n_counts"]), bins=100)
_ = plt.xlabel("log10 Counts Per cell")
_ = plt.ylabel("# Cells")


"""
count_adat_fn = 'example_PBMC/counts.h5ad'
sc.write(count_adat_fn, adata)               
"""

numhvgenes = (
    2000  ## Number of over-dispersed genes to use for running the actual factorizations
)

## Specify the Ks to use as a space separated list in this case "5 6 7 8 9 10"
K = " ".join([str(i) for i in range(5, 11)])
print(K)

adata.X = np.array(adata.X.todense())
tpm = nmh.compute_tpm(adata)
norm_counts = nmh.get_norm_counts(adata, tpm, num_highvar_genes=numhvgenes)

from cv_nmf import my_plot_nmf

my_plot_nmf(
    norm_counts.X,
    truth=7,
    replicates=5,
    k0=5,
    k=11,
    fig_name="./nmf_cv_curve_pbmc_data_r_5_k_5_11_run_1.pdf",
)
