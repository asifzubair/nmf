#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 14:38:11 2021

@author: asif
"""

import argparse as ap
import os
import sys

import pandas as pd
import scanpy as sc

sys.path.append("../cNMF")
import nmf.nmf_helpers as nmh
from nmf.cv_nmf import plot_cv_nmf, run_cv_nmf, save_cv_nmf
from nmf.nmf_helpers import load_df_from_npz


def get_args():
    parser = ap.ArgumentParser(description="Run CV NMF on simulated data")
    parser.add_argument("path", help="location of simulation directory")
    parser.add_argument(
        "--save", default="./nmf_cv_curve.pdf", help="save filename path"
    )
    parser.add_argument(
        "--pholdout", default=0.3, type=float, help="fraction of cells in test set"
    )
    parser.add_argument(
        "--k0", default=2, type=int, help="lower bound for number of factors"
    )
    parser.add_argument(
        "--k", default=15, type=int, help="upper bound for number of factors"
    )
    parser.add_argument(
        "--replicates", default=1, type=int, help="number of replicates"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    path = args.path
    save_fname = args.save
    k0 = args.k0
    k = args.k
    pholdout = args.pholdout
    replicates = args.replicates

    counts_file = os.path.join(path, "counts.npz")
    counts = load_df_from_npz(counts_file)
    input_counts = sc.AnnData(
        X=counts.values,
        obs=pd.DataFrame(index=counts.index),
        var=pd.DataFrame(index=counts.columns),
    )
    tpm = nmh.compute_tpm(input_counts)
    norm_counts = nmh.get_norm_counts(input_counts, tpm, num_highvar_genes=2000)

    print("Shape of input matrix: {}".format(norm_counts.X.shape))

    cv_out = run_cv_nmf(
        norm_counts.X, k0=k0, k=k, replicates=replicates, p_holdout=pholdout
    )
    save_cv_nmf(
        *cv_out,
        fname=os.path.join(
            path, f"cv_output_k0_{k0}_k_{k}_holdout_{pholdout}_replicates_{replicates}"
        ),
    )

    plot_cv_nmf(*cv_out, fig_name=save_fname)
