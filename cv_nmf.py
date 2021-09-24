#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:05:13 2021

@author: asif
"""

import numpy as np
from numpy.random import randn, rand
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from nnls import nnlsm_blockpivot as nnlstsq
import itertools
from scipy.spatial.distance import cdist
from tqdm.contrib.itertools import product
import logging

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B

    Note: uses a broadcasted solve for speed.

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T
    except:
        r = T.shape[1]
        T[:,np.arange(r),np.arange(r)] += 1e-6
        return np.squeeze(np.linalg.solve(T, rhs), axis=-1).T

def censored_lstsq_slow(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    X = np.empty((A.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        m = M[:,i] # drop rows where mask is zero
        X[:,i] = np.linalg.lstsq(A[m], B[m,i])[0]
    return X

def censored_nnlstsq(A, B, M):
    """Solves nonnegative least-squares problem with missing data in B

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : nonnegative r x n matrix that minimizes norm(M*(AX - B))
    """
    if A.ndim == 1:
        A = A[:,None]
    rhs = np.dot(A.T, M * B).T[:,:,None] # n x r x 1 tensor
    T = np.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    X = np.empty((B.shape[1], A.shape[1]))
    for n in range(B.shape[1]):
        X[n] = nnlstsq(T[n], rhs[n], is_input_prod=True)[0].T
    return X.T

def censored_nnlstsq_slow(A, B, M):
    """Solves least squares problem subject to missing data.

    Note: uses a for loop over the columns of B, leading to a
    slower but more numerically stable algorithm

    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)

    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """

    X = np.empty((A.shape[1], B.shape[1]))
    for i in range(B.shape[1]):
        m = M[:,i] # drop rows where mask is zero
        X[:,i] = nnlstsq(A[m], B[m,i], is_input_prod=True)[0].T
    return X

def cv_pca(data, rank, M=None, p_holdout=0.3, nonneg=False):
    """Fit PCA or NMF while holding out a fraction of the dataset.
    """

    # choose solver for alternating minimization
    if nonneg:
        solver = censored_nnlstsq
    else:
        solver = censored_lstsq

    # create masking matrix
    if M is None:
        M = np.random.rand(*data.shape) > p_holdout

    # initialize U randomly
    if nonneg:
        U = np.random.rand(data.shape[0], rank)
    else:
        U = np.random.randn(data.shape[0], rank)

    # fit pca/nmf
    for itr in range(100):
        Vt = solver(U, data, M)
        U = solver(Vt.T, data.T, M.T).T

    # return result and test/train error
    resid = np.dot(U, Vt) - data
    train_err = np.mean(resid[M]**2)
    test_err = np.mean(resid[~M]**2)
    return U, Vt, train_err, test_err


def cv_kmeans(data, rank, p_holdout=.3, M=None):
    """Fit kmeans while holding out a fraction of the dataset.
    """

    # create masking matrix
    if M is None:
        M = np.random.rand(*data.shape) > p_holdout

    # initialize cluster centers
    Vt = np.random.randn(rank, data.shape[1])
    U = np.empty((data.shape[0], rank))
    rn = np.arange(U.shape[0])

    # initialize missing data randomly
    imp = data.copy()
    imp[~M] = np.random.randn(*data.shape)[~M]

    # initialize cluster centers far apart
    Vt = [imp[np.random.randint(data.shape[0])]]
    while len(Vt) < rank:
        i = np.argmax(np.min(cdist(imp, Vt), axis=1))
        Vt.append(imp[i])
    Vt = np.array(Vt)

    # fit kmeans
    for itr in range(50):
        # update cluster assignments
        clus = np.argmin(cdist(imp, Vt), axis=1)
        U.fill(0.0)
        U[rn, clus] = 1.0
        # update centroids
        Vt = censored_lstsq(U, imp, M)
        assert np.all(np.sum(np.abs(Vt), axis=1) > 0)
        # update estimates of missing data
        imp[~M] = np.dot(U, Vt)[~M]

    # return result and test/train error
    resid = np.dot(U, Vt) - data
    train_err = np.mean(resid[M]**2)
    test_err = np.mean(resid[~M]**2)
    return clus, U, Vt, train_err, test_err


def plot_pca():
    # parameters
    N, R = 150, 4
    noise = 2
    replicates = 1
    ranks = np.arange(1, 8)

    # initialize
    U = np.random.randn(N, R)
    Vt = np.random.randn(R, N)
    data = np.dot(U, Vt) + noise*np.random.randn(N, N)
    train_err, test_err = [], []

    # fit models
    for rnk, _ in itertools.product(ranks, range(replicates)):
        tr, te = cv_pca(data, rnk)[2:]
        train_err.append((rnk, tr))
        test_err.append((rnk, te))

    # make plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
    ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Number of PCs')
    ax.set_title('PCA')
    ax.axvline(4, color='k', dashes=[2,2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig('../../img/pca-crossval/pca_cv_curve.pdf')

def plot_nmf():
    # parameters
    N, R = 150, 4
    noise = .8
    replicates = 1
    ranks = np.arange(1, 8)

    # initialize problem
    U = np.random.rand(N, R)
    Vt = np.random.rand(R, N)
    data = np.dot(U, Vt) + noise*np.random.rand(N, N)
    train_err, test_err = [], []

    # fit models
    for rnk, _ in itertools.product(ranks, range(replicates)):
        tr, te = cv_pca(data, rnk, nonneg=True)[2:]
        train_err.append((rnk, tr))
        test_err.append((rnk, te))

    # make plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
    ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Number of factors')
    ax.set_title('NMF')
    ax.axvline(4, color='k', dashes=[2,2])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig('../../img/pca-crossval/nmf_cv_curve.pdf')

def my_plot_pca(data, truth=7, replicates=1, k0=1, k = 8, p_holdout=0.3, fig_name="./nmf_cv_curve.pdf"):
    """
    # parameters
    N, R = 150, 4
    noise = .8
    """
    replicates = replicates
    ranks = np.arange(k0, k)

    """
    # initialize problem
    U = np.random.rand(N, R)
    Vt = np.random.rand(R, N)
    data = np.dot(U, Vt) + noise*np.random.rand(N, N)
    """
    train_err, test_err, rr = [], [], []

    # fit models
    for rnk, _ in itertools.product(ranks, range(replicates)):
        print(rnk)
        tr, te = cv_pca(data, rnk, p_holdout=p_holdout)[2:]
        train_err.append(tr)
        test_err.append(te)
        rr.append(rnk)
        
    rr = np.array(rr)
    train_err, test_err = np.array(train_err), np.array(test_err)
    mean_train = [np.mean(train_err[rr==rnk]) for rnk in ranks]
    mean_test = [np.mean(test_err[rr==rnk]) for rnk in ranks]
    
    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].plot(ranks, mean_train, '-b', label='Train Data')
    ax[1].plot(ranks, mean_test, '-r', label='Test Data')
#    ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
#    ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].set_xlabel('Number of factors')
    ax[1].set_xlabel('Number of factors')
    ax[0].set_title('PCA')
    ax[1].axvline(truth, color='k', dashes=[2,2])
    ax[0].plot(rr, train_err, 'ob', alpha=.5, ms=3, mec=None)
    ax[1].plot(rr, test_err, 'or', alpha=.5, ms=3, mec=None)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend()
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend()   
    fig.tight_layout()
    fig.savefig(fig_name)


def run_cv_nmf(data, replicates=1, k0=2, k=15, p_holdout=0.3):
    replicates = replicates
    ranks = np.arange(k0, k+1)
    train_err, test_err, rr = [], [], []

    # fit models
    for rnk, _ in product(ranks, range(replicates)):
#        print(rnk)
        tr, te = cv_pca(data, rnk, p_holdout=p_holdout, nonneg=True)[2:]
        train_err.append(tr)
        test_err.append(te)
        rr.append(rnk)
        
    rr = np.array(rr)
    train_err, test_err = np.array(train_err), np.array(test_err)
        
    return ranks, rr, train_err, test_err


from multiprocessing import Pool
from functools import partial

def par_cv_nmf(rank, data, p_holdout):
    return cv_pca(data=data, rank=rank, p_holdout=p_holdout, nonneg=True)[2:]

def run_par_cv_nmf(data, replicates=1, k0=2, k=15, p_holdout=0.3, num_processors=16):
    replicates = replicates
    ranks = np.arange(k0, k+1)
    train_err, test_err, rr = [], [], []

    ks = [k[0] for k in product(ranks, range(replicates))]
    
    logging.debug(f"starting computations on {num_processors} cores")
    
    # fit models
    with Pool(processes=num_processors) as pool:
        out = pool.map(partial(par_cv_nmf, data=data, p_holdout=p_holdout), ks)

    for (tr, te), rnk in zip(out, ks):
        train_err.append(tr)
        test_err.append(te)
        rr.append(rnk)
        
    rr = np.array(rr)
    train_err, test_err = np.array(train_err), np.array(test_err)
        
    return ranks, rr, train_err, test_err


def save_cv_nmf(ranks, rr, train_err, test_err, fname):
    np.savez_compressed(fname, ranks=ranks, rr=rr, train_err=train_err, test_err=test_err)


def plot_cv_nmf(ranks, rr, train_err, test_err, fig_name, truth = None):
    mean_train = [np.mean(train_err[rr==rnk]) for rnk in ranks]
    mean_test = [np.mean(test_err[rr==rnk]) for rnk in ranks]
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].plot(ranks, mean_train, '-b', label='Train Data')
    ax[1].plot(ranks, mean_test, '-r', label='Test Data')
#    ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
#    ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].set_xlabel('Number of factors')
    ax[1].set_xlabel('Number of factors')
    ax[0].set_title('NMF')
    if truth:
        ax[1].axvline(truth, color='k', dashes=[2,2])
    ax[0].plot(rr, train_err, 'ob', alpha=.5, ms=3, mec=None)
    ax[1].plot(rr, test_err, 'or', alpha=.5, ms=3, mec=None)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend()
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend()   
    fig.tight_layout()
    fig.savefig(fig_name)


def my_plot_nmf(data, truth=7, replicates=1, k0=1, k = 8, p_holdout=0.3, fig_name="./nmf_cv_curve.pdf"):
    """
    # parameters
    N, R = 150, 4
    noise = .8
    """
    replicates = replicates
    ranks = np.arange(k0, k)

    """
    # initialize problem
    U = np.random.rand(N, R)
    Vt = np.random.rand(R, N)
    data = np.dot(U, Vt) + noise*np.random.rand(N, N)
    """
    train_err, test_err, rr = [], [], []

    # fit models
    for rnk, _ in itertools.product(ranks, range(replicates)):
        print(rnk)
        tr, te = cv_pca(data, rnk, p_holdout=p_holdout, nonneg=True)[2:]
        train_err.append(tr)
        test_err.append(te)
        rr.append(rnk)
        
    rr = np.array(rr)
    train_err, test_err = np.array(train_err), np.array(test_err)
    mean_train = [np.mean(train_err[rr==rnk]) for rnk in ranks]
    mean_test = [np.mean(test_err[rr==rnk]) for rnk in ranks]
    
    # make plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    ax[0].plot(ranks, mean_train, '-b', label='Train Data')
    ax[1].plot(ranks, mean_test, '-r', label='Test Data')
#    ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
#    ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
    ax[0].set_ylabel('Mean Squared Error')
    ax[0].set_xlabel('Number of factors')
    ax[1].set_xlabel('Number of factors')
    ax[0].set_title('NMF')
    ax[1].axvline(truth, color='k', dashes=[2,2])
    ax[0].plot(rr, train_err, 'ob', alpha=.5, ms=3, mec=None)
    ax[1].plot(rr, test_err, 'or', alpha=.5, ms=3, mec=None)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend()
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend()   
    fig.tight_layout()
    fig.savefig(fig_name)


def plot_kmeans():
    # parameters
    N, R = 150, 4
    noise = 1.5
    ranks = np.arange(1, 8)
    replicates = 10

    # initialize problem
    U = np.zeros((N, R))
    U[np.arange(N), np.random.randint(R, size=N)] = 1
    Vt = np.random.randn(R, N)
    data = np.dot(U, Vt) + noise*np.random.randn(N, N)
    train_err, test_err, rr = [], [], []

    # fit models
    for rnk, _ in itertools.product(ranks, range(replicates)):
        tr, te = cv_kmeans(data, rnk)[3:]
        train_err.append(tr)
        test_err.append(te)
        rr.append(rnk)

    rr = np.array(rr)
    train_err, test_err = np.array(train_err), np.array(test_err)
    mean_train = [np.mean(train_err[rr==rnk]) for rnk in ranks]
    mean_test = [np.mean(test_err[rr==rnk]) for rnk in ranks]

    # make plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
    ax.plot(ranks, mean_train, '-b', label='Train Data')
    ax.plot(ranks, mean_test, '-r', label='Test Data')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Number of clusters')
    ax.set_title('K-means clustering')
    ax.axvline(4, color='k', dashes=[2,2])
    ax.plot(rr, train_err, 'ob', alpha=.5, ms=3, mec=None)
    ax.plot(rr, test_err, 'or', alpha=.5, ms=3, mec=None)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    fig.tight_layout()
    fig.savefig('../../img/pca-crossval/kmeans_cv_curve.pdf')





if __name__ == '__main__':
    plot_pca()
    plot_nmf()
    plot_kmeans()
    plt.show()