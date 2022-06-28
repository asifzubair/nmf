#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:56:23 2021

@author: asif
"""

import typing as t

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scanpy import AnnData
from sklearn.decomposition import non_negative_factorization
from sklearn.utils import sparsefuncs


def compute_tpm(input_counts):
    """
    Default TPM normalization
    """
    tpm = input_counts.copy()
    sc.pp.normalize_per_cell(tpm, counts_per_cell_after=1e6)
    return tpm


def load_df_from_npz(filename):
    """load dataframe from Numpy pickle"""
    with np.load(filename, allow_pickle=True) as f:
        obj = pd.DataFrame(**f)
    return obj


def var_sparse_matrix(X):
    """compute variance of genes for sparse matrix"""
    mean = np.array(X.mean(axis=0)).reshape(-1)
    Xcopy = X.copy()
    Xcopy.data **= 2
    var = np.array(Xcopy.mean(axis=0)).reshape(-1) - (mean**2)
    return var


def get_highvar_genes(
    input_counts, expected_fano_threshold=None, minimal_mean=0.5, numgenes=None
):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var / gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_counts_var) / gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = (
        (gene_counts_fano > w_fano_low)
        & (gene_counts_fano < w_fano_high)
        & (gene_counts_mean > w_mean_low)
        & (gene_counts_mean < w_mean_high)
    )
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2) * gene_counts_mean + (B**2)

    fano_ratio = gene_counts_fano / gene_expected_fano

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T = None
    else:
        if not expected_fano_threshold:
            T = 1.0 + gene_counts_fano[winsor_box].std()
        else:
            T = expected_fano_threshold
        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame(
        {
            "mean": gene_counts_mean,
            "var": gene_counts_var,
            "fano": gene_counts_fano,
            "expected_fano": gene_expected_fano,
            "high_var": high_var_genes_ind,
            "fano_ratio": fano_ratio,
        }
    )
    gene_fano_parameters = {
        "A": A,
        "B": B,
        "T": T,
        "minimal_mean": minimal_mean,
    }
    return (gene_counts_stats, gene_fano_parameters)


def get_norm_counts(
    counts: AnnData,
    tpm: AnnData,
    high_variance_genes_filter: np.array = None,
    num_highvar_genes: int = None,
) -> AnnData:

    if high_variance_genes_filter is None:
        ## Get list of high-var genes if one wasn't provided
        if sp.issparse(tpm.X):
            #            (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(tpm.X, numgenes=num_highvar_genes)
            raise NotImplementedError
        else:
            (gene_counts_stats, gene_fano_params) = get_highvar_genes(
                np.array(tpm.X), numgenes=num_highvar_genes
            )

        high_variance_genes_filter = list(
            tpm.var.index[gene_counts_stats.high_var.values]
        )
    ## Subset out high-variance genes
    norm_counts = counts[:, high_variance_genes_filter]

    ## Scale genes to unit variance
    if sp.issparse(tpm.X):
        raise NotImplementedError
    #       sc.pp.scale(norm_counts, zero_center=False)
    #       if np.isnan(norm_counts.X.data).sum() > 0:
    #           print('Warning NaNs in normalized counts matrix')
    else:
        norm_counts.X /= norm_counts.X.std(axis=0, ddof=1)
        if np.isnan(norm_counts.X).sum().sum() > 0:
            print("Warning NaNs in normalized counts matrix")

        ## Save a \n-delimited list of the high-variance genes used for factorization
    #        open(self.paths['nmf_genes_list'], 'w').write('\n'.join(high_variance_genes_filter))

    ## Check for any cells that have 0 counts of the overdispersed genes
    zerocells = norm_counts.X.sum(axis=1) == 0
    if zerocells.sum() > 0:
        examples = norm_counts.obs.index[zerocells]
        print(
            "Warning: %d cells have zero counts of overdispersed genes. E.g. %s"
            % (zerocells.sum(), examples[0])
        )
        print("Consensus step may not run when this is the case")

    return norm_counts


def nmf_factorize(input_counts, n_components=10, random_state=124578):
    """factorize using the cNMF tools"""
    input_counts = sc.AnnData(
        X=input_counts.values,
        obs=pd.DataFrame(index=input_counts.index),
        var=pd.DataFrame(index=input_counts.columns),
    )

    tpm = compute_tpm(input_counts)
    norm_counts = get_norm_counts(input_counts, tpm, num_highvar_genes=1500)
    if norm_counts.X.dtype != np.float64:
        norm_counts.X = norm_counts.X.astype(np.float64)

    nmf_kwargs = dict(
        alpha=0.0,
        l1_ratio=0.0,
        beta_loss="frobenius",  # "kullback-leibler"
        solver="cd",  # "mu"
        tol=1e-4,
        max_iter=1000,
        regularization=None,
        init="random",
    )
    nmf_kwargs["n_components"] = n_components
    nmf_kwargs["random_state"] = random_state

    (usages, spectra, niter) = non_negative_factorization(norm_counts.X, **nmf_kwargs)
    return usages, spectra, niter
