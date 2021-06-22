#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:48:08 2021

@author: asif
"""

"""
We try to examine behaviour of the CV NMF algorithm as it is implemented.

We will use a matrix of size (6442, 2000) to find the optimal value of K
"""

import os
import argparse as ap
import numpy as np

#from numpy.random import default_rng
#rng = default_rng()

from cv_nmf import run_par_cv_nmf, save_cv_nmf, plot_cv_nmf

import logging
log = logging.getLogger()

console = logging.StreamHandler()
log.addHandler(console)
log.setLevel(logging.INFO)

def get_args():
    parser = ap.ArgumentParser(description= "Run CV NMF on basic simulated data")
    parser.add_argument('--c', default=16, type=int,
                        help='number of processors to use')
    parser.add_argument('--k', default=25, type=int,
                        help='truth value for the simulation [25]')
    parser.add_argument('--k0', default=5, type=int,
                        help='lower bound for number of factors [5]')
    parser.add_argument('--k1', default=50, type=int,
                        help='upper bound for number of factors [50]')
    parser.add_argument('--replicates', default=1, type=int, 
                        help='number of replicates [1]')
    parser.add_argument('--pholdout', default=0.3, type=float, 
                        help='fraction of cells in test set [0.3]')
    parser.add_argument('--save-dir', default='./sims',
                        help='save filename path [./sims]')
    parser.add_argument('--m', default=6442, type=int,
                        help='number of rows of input matrix [6442]')
    parser.add_argument('--n', default=6442, type=int,
                        help='number of columns of input matrix [2000]')
    parser.add_argument('--noise', default=0.8, type=float,
                        help='noise amplitude [0.8]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    k0 = args.k0
    k1 = args.k1
    K = args.k
    replicates = args.replicates
    p_holdout = args.pholdout
    num_processors = args.c
    save_dir = args.save_dir
    cv_save = f"Sims_k0_{k0}_k_{k1}_holdout_{p_holdout}_replicates_{replicates}_truth_{K}"
    
    M = args.m
    N = args.n 
    noise = args.noise

#    A = rng.negative_binomial(10, 0.8, (M, K))
#    B = rng.negative_binomial(10, 0.8, (K, N))
#    E = rng.standard_exponential((M, N))
    
    A = np.random.rand(M, K)
    B = np.random.rand(K, N)
    E = np.random.rand(M, N)

    assert np.linalg.matrix_rank(A) == K
    assert np.linalg.matrix_rank(B) == K

    C = (A @ B) + noise*E
    rankC = np.linalg.matrix_rank(C) 
    assert rankC == min(M, N)

    log.info(f"Input matrix is of size {C.shape} and of rank {rankC}, truth value is {K}, noise amplitude is {noise}")

    os.mkdir(save_dir)
    cv_out = run_par_cv_nmf(C, k0=k0, k=k1, replicates=replicates, p_holdout=p_holdout, num_processors=num_processors) 
    save_cv_nmf(*cv_out, fname=os.path.join(save_dir, cv_save))
    
    save_fname = os.path.join(save_dir, cv_save + ".pdf")
    plot_cv_nmf(*cv_out, fig_name=save_fname, truth=K)
   