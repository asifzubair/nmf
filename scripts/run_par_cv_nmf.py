#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 11:29:35 2021

@author: asif
"""

import argparse as ap
import os
import sys

import pandas as pd
import scanpy as sc

sys.path.append("../cNMF")
import logging

import nmf_helpers as nmh
import pyreadr
from cnmf import load_df_from_npz
from cv_nmf import plot_cv_nmf, run_par_cv_nmf, save_cv_nmf

logging.basicConfig(level=logging.DEBUG)


def get_args():
    parser = ap.ArgumentParser(description="Run CV NMF on simulated splatter data")
    parser.add_argument("path", help="location of simulation directory")
    parser.add_argument("--r", action="store_true", help="read from an RDS file")
    parser.add_argument("--n", default=16, type=int, help="number of processors to use")
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
    parser.add_argument(
        "--save", default="./cv_nmf_curve.pdf", help="save filename path"
    )
    args = parser.parse_args()

    if args.r and not os.path.splitext(args.path)[1].lower() == ".rds":
        print("Please provide RDS file with flag 'r'.\nExiting ...")
        sys.exit(0)
    return args


if __name__ == "__main__":
    args = get_args()
    path = args.path
    save_fname = args.save
    k0 = args.k0
    k = args.k
    p_holdout = args.pholdout
    replicates = args.replicates
    use_r = args.r
    nprocessors = args.n

    if use_r:
        counts = pyreadr.read_r(path)[None]
        path, _ = os.path.splitext(path)
        rds_prefix = os.path.basename(path)
        path = os.path.dirname(path)
    else:
        counts_file = os.path.join(path, "counts.npz")
        counts = load_df_from_npz(counts_file)

    input_counts = sc.AnnData(
        X=counts.values,
        obs=pd.DataFrame(index=counts.index),
        var=pd.DataFrame(index=counts.columns),
    )
    tpm = nmh.compute_tpm(input_counts)
    norm_counts = nmh.get_norm_counts(input_counts, tpm, num_highvar_genes=2000)

    logging.debug("Shape of input matrix: {}".format(norm_counts.X.shape))

    cv_out = run_par_cv_nmf(
        norm_counts.X,
        k0=k0,
        k=k,
        replicates=replicates,
        p_holdout=p_holdout,
        num_processors=nprocessors,
    )

    if use_r:
        save_cv_nmf(
            *cv_out,
            fname=os.path.join(
                path,
                f"{rds_prefix}_k0_{k0}_k_{k}_holdout_{p_holdout}_replicates_{replicates}",
            ),
        )
    else:
        save_cv_nmf(
            *cv_out,
            fname=os.path.join(
                path,
                f"cv_output_k0_{k0}_k_{k}_holdout_{p_holdout}_replicates_{replicates}",
            ),
        )

    plot_cv_nmf(*cv_out, fig_name=save_fname)
