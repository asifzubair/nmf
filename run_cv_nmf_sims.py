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


import numpy as np
import logging
import os

log = logging.getLogger()

console = logging.StreamHandler()
log.addHandler(console)
log.setLevel(logging.INFO)

K = 25
A = np.random.rand(6442, K)
B = np.random.rand(K, 2000)

assert np.linalg.matrix_rank(A) == K
assert np.linalg.matrix_rank(B) == K

C = A @ B

log.info(f"Input matrix is of size {C.shape}")

from cv_nmf import run_par_cv_nmf, save_cv_nmf, plot_cv_nmf

path = "./data/np_sims"
k0 = 5
k = 50
replicates = 5
p_holdout = 0.1
num_processors = 12
cv_save = f"Sims_k0_{k0}_k_{k}_holdout_{p_holdout}_replicates_{replicates}"

cv_out = run_par_cv_nmf(C, k0=5, k=13, replicates=1, p_holdout=0.1, num_processors=12) 
save_cv_nmf(*cv_out, fname=os.path.join(path, cv_save))

save_fname = cv_save + ".pdf"
plot_cv_nmf(*cv_out, fig_name=save_fname)