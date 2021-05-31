#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:56:23 2021

@author: asif
"""

import typing as t

import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.utils import sparsefuncs
from sklearn.decomposition import non_negative_factorization

import scanpy as sc
from scanpy import AnnData

def compute_tpm(input_counts):
    """
    Default TPM normalization
    """
    tpm = input_counts.copy()
    sc.pp.normalize_per_cell(tpm, counts_per_cell_after=1e6)
    return(tpm)


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
    return(var)


def get_highvar_genes(input_counts, expected_fano_threshold=None,
                       minimal_mean=0.5, numgenes=None):
    # Find high variance genes within those cells
    gene_counts_mean = pd.Series(input_counts.mean(axis=0).astype(float))
    gene_counts_var = pd.Series(input_counts.var(ddof=0, axis=0).astype(float))
    gene_counts_fano = pd.Series(gene_counts_var/gene_counts_mean)

    # Find parameters for expected fano line
    top_genes = gene_counts_mean.sort_values(ascending=False)[:20].index
    A = (np.sqrt(gene_counts_var)/gene_counts_mean)[top_genes].min()

    w_mean_low, w_mean_high = gene_counts_mean.quantile([0.10, 0.90])
    w_fano_low, w_fano_high = gene_counts_fano.quantile([0.10, 0.90])
    winsor_box = ((gene_counts_fano > w_fano_low) &
                    (gene_counts_fano < w_fano_high) &
                    (gene_counts_mean > w_mean_low) &
                    (gene_counts_mean < w_mean_high))
    fano_median = gene_counts_fano[winsor_box].median()
    B = np.sqrt(fano_median)

    gene_expected_fano = (A**2)*gene_counts_mean + (B**2)

    fano_ratio = (gene_counts_fano/gene_expected_fano)

    # Identify high var genes
    if numgenes is not None:
        highvargenes = fano_ratio.sort_values(ascending=False).index[:numgenes]
        high_var_genes_ind = fano_ratio.index.isin(highvargenes)
        T=None
    else:
        if not expected_fano_threshold:
            T = (1. + gene_counts_fano[winsor_box].std())
        else:
            T = expected_fano_threshold
        high_var_genes_ind = (fano_ratio > T) & (gene_counts_mean > minimal_mean)

    gene_counts_stats = pd.DataFrame({
        'mean': gene_counts_mean,
        'var': gene_counts_var,
        'fano': gene_counts_fano,
        'expected_fano': gene_expected_fano,
        'high_var': high_var_genes_ind,
        'fano_ratio': fano_ratio
        })
    gene_fano_parameters = {
            'A': A, 'B': B, 'T':T, 'minimal_mean': minimal_mean,
        }
    return(gene_counts_stats, gene_fano_parameters)


def get_norm_counts(counts: AnnData, tpm: AnnData,
                    high_variance_genes_filter: np.array = None,
                    num_highvar_genes: int = None) -> AnnData:
    
    if high_variance_genes_filter is None:
        ## Get list of high-var genes if one wasn't provided
        if sp.issparse(tpm.X):
#            (gene_counts_stats, gene_fano_params) = get_highvar_genes_sparse(tpm.X, numgenes=num_highvar_genes)
            raise NotImplementedError
        else:
            (gene_counts_stats, gene_fano_params) = get_highvar_genes(np.array(tpm.X), numgenes=num_highvar_genes)
                
        high_variance_genes_filter = list(tpm.var.index[gene_counts_stats.high_var.values])
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
            print('Warning NaNs in normalized counts matrix')                    
        
        ## Save a \n-delimited list of the high-variance genes used for factorization
#        open(self.paths['nmf_genes_list'], 'w').write('\n'.join(high_variance_genes_filter))

    ## Check for any cells that have 0 counts of the overdispersed genes
    zerocells = norm_counts.X.sum(axis=1)==0
    if zerocells.sum()>0:
        examples = norm_counts.obs.index[zerocells]
        print('Warning: %d cells have zero counts of overdispersed genes. E.g. %s' % (zerocells.sum(), examples[0]))
        print('Consensus step may not run when this is the case')
        
    return(norm_counts)


def nmf_factorize(input_counts, n_components = 10, random_state = 124578):
    """ factorize using the cNMF tools """
    input_counts = sc.AnnData(X=input_counts.values,
                          obs=pd.DataFrame(index=input_counts.index),
                          var=pd.DataFrame(index=input_counts.columns))
    
    tpm = compute_tpm(input_counts)
    norm_counts = get_norm_counts(input_counts, tpm, num_highvar_genes = 1500)
    if norm_counts.X.dtype != np.float64:
        norm_counts.X = norm_counts.X.astype(np.float64)
    
    nmf_kwargs = dict( alpha = 0.0,
                      l1_ratio = 0.0, 
                      beta_loss = "frobenius", # "kullback-leibler"
                      solver = "cd", # "mu"
                      tol = 1e-4,
                      max_iter = 1000, 
                      regularization = None, 
                      init = "random")
    nmf_kwargs["n_components"] = n_components
    nmf_kwargs["random_state"] = random_state

    (usages, spectra, niter) = non_negative_factorization(norm_counts.X, **nmf_kwargs)
    return usages, spectra, niter


def split_molecules(
    umis: np.ndarray,
    data_split: float,
    overlap_factor: float = 0.0,
    random_state: np.random.RandomState = None,
) -> t.Tuple[np.ndarray, np.ndarray]:
    """Splits molecules into two (potentially overlapping) groups.
    :param umis: Array of molecules to split
    :param data_split: Proportion of molecules to assign to the first group
    :param overlap_factor: Overlap correction factor, if desired
    :param random_state: For reproducible sampling
    :return: umis_X and umis_Y, representing ``split`` and ``~(1 - split)`` counts
             sampled from the input array
    https://github.com/czbiohub/molecular-cross-validation/blob/04d9df0e59a247200153b3ce38952a086169769f/src/molecular_cross_validation/util.py#L235
    """
    if random_state is None:
        random_state = np.random.RandomState()

    umis_X_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_Y_disjoint = random_state.binomial(
        umis - umis_X_disjoint, (1 - data_split) / (1 - data_split + overlap_factor)
    )
    overlap_factor = umis - umis_X_disjoint - umis_Y_disjoint
    umis_X = umis_X_disjoint + overlap_factor
    umis_Y = umis_Y_disjoint + overlap_factor

    return umis_X, umis_Y


def convert_expectations(
    exp_values: np.ndarray,
    expected_func: t.Callable,
    max_val: float,
    a: t.Union[float, np.ndarray],
    b: t.Union[float, np.ndarray] = None,
) -> np.ndarray:
    """Given an estimate of the mean of f(X) where X is a Poisson random variable, this
    function will scale those estimates from scale ``a`` to ``b`` by using the function
    ``expected_func`` to calculate a grid of values over the relevant range defined by
    ``[0, max_val]``. Used by the methods below for scaling sqrt count and log1p counts.
    :param exp_values: The estimated mean of f(X) calculated at scale ``a``
    :param expected_func: Function to esimate E[f(X)] from a Poisson mean.
    :param max_val: Largest count relevant for computing interpolation. Using a very
        high value will use more memory but is otherwise harmless.
    :param a: Scaling factor(s) of the input data
    :param b: Scale for the output. Set to ``1 - a`` by default
    :return: A scaled array of mean expression values
    https://github.com/czbiohub/molecular-cross-validation/blob/04d9df0e59a247200153b3ce38952a086169769f/src/molecular_cross_validation/util.py#L87
    """
    if b is None:
        b = 1.0 - a

    # this code creates a grid of values for computing the interpolation arrays. We use
    # exponentially growing space between points to save memory
    vs = 2 ** np.arange(0, np.ceil(np.log2(max_val + 1)) + 1) - 1
    p_range = np.hstack(
        [np.arange(v, vs[i + 1], 2 ** (i + 1) * 0.01) for i, v in enumerate(vs[:-1])]
    )

    xp = expected_func(p_range * a)
    fp = expected_func(p_range * b)

    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        xp = np.broadcast_to(xp, (exp_values.shape[0], p_range.shape[0]))
        fp = np.broadcast_to(fp, (exp_values.shape[0], p_range.shape[0]))

        interps = np.empty_like(exp_values)
        for i in range(exp_values.shape[0]):
            interps[i, :] = np.interp(exp_values[i, :], xp[i, :], fp[i, :])

        return interps
    else:
        return np.interp(exp_values, xp, fp)        