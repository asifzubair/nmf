#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:43:22 2021

@author: asif
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
#plt.style.use('../../matplotlibrc')
import numpy as np
import seaborn as sns
import sys
#from bhtsne import tsne
import glob, os

sys.path.append('../cNMF')
from cnmf import get_highvar_genes, save_df_to_npz, load_df_from_npz
from plotting import make_genestats_plot

"""
cluster_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE102nnn/GSE102827/suppl/GSE102827_cell_type_assignments.csv.gz'
cluster_fn = '../../../data/Part3_VisualCortex/GSE102827_cell_type_assignments.csv.gz'

matrix_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE102nnn/GSE102827/suppl/GSE102827_merged_all_raw.csv.gz'
matrix_fn = '../../../data/Part3_VisualCortex/GSE102827_merged_all_raw.csv.gz'


if not os.path.exists('./data/Part3_VisualCortex'):
    os.mkdir('./data/Part3_VisualCortex')

cmd = 'wget -O %s %s; gzip -f -d %s'% (cluster_fn, cluster_url, cluster_fn)
print(cmd)
os.system(cmd)

cmd = 'wget -O %s %s; gzip -f -d %s'% (matrix_fn, matrix_url, matrix_fn)
print(cmd)
!{cmd}
"""

##
## Save the counts matrix in the compressed npz format
##

counts_fn = './data/Part3_VisualCortex/GSE102827_merged_all_raw.csv'
counts = pd.read_csv(counts_fn, sep=',', index_col=0).T
save_df_to_npz(counts, './data/Part3_VisualCortex/GSE102827_merged_all_raw.npz')


##
## Keep only the neurons (defined based on the previous clustering) and exclude the other cell-types
##

counts = load_df_from_npz('./data/Part3_VisualCortex/GSE102827_merged_all_raw.npz')

celltype = pd.read_csv('./data/Part3_VisualCortex/GSE102827_cell_type_assignments.csv', index_col=0)
celltype['maintype'].value_counts()

neuron_ind = celltype.loc[counts.index, 'maintype'].isin(['Excitatory', 'Interneurons'])
non_neuron_cells = neuron_ind.index[~neuron_ind]
counts.drop(non_neuron_cells, axis=0, inplace=True)

save_df_to_npz(counts, './data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_raw.npz')


##
## Drop low count genes and cells (cells with fewer than 1000 UMIs, genes that are detected in fewer than 1/500 cells)
##


(fig,ax) = plt.subplots(1,1, figsize=(2,2), dpi=300)
countcutoff=3.0

counts_per_cell = counts.sum(axis=1)
counts.drop(counts_per_cell.index[counts_per_cell==0], axis=0, inplace=True)
counts_per_cell = counts_per_cell.loc[counts_per_cell>0]


ax.hist(counts_per_cell.apply(np.log10), bins=90)
_ = ax.set_title('Log10 Counts Per Cell')
lims = ax.get_ylim()
ax.vlines(x=countcutoff, ymin=0, ymax=lims[1], linestyle='--', label='minimum threshold')
_ = ax.set_ylim(lims)

TPM = counts.div(counts.sum(axis=1), axis=0) * (10**6)


nnzthresh = counts.shape[0] / 500
numnonzero = (counts>0).sum(axis=0)
print((numnonzero>nnzthresh).value_counts())
ind = numnonzero<200
(fig,ax) = plt.subplots(1,1, figsize=(2,2), dpi=300)
_ = ax.hist(numnonzero.loc[ind], bins=100)
(_,ymax) = ax.get_ylim()
ax.vlines(x=nnzthresh, ymin=0, ymax=ymax, linestyle='--')
ax.set_xlabel('# Samples With Non-zero Count')
ax.set_ylabel('# Genes')
              
genestodrop = numnonzero.index[(numnonzero<=nnzthresh)]
counts.drop(genestodrop, axis=1, inplace=True)
TPM.drop(genestodrop, axis=1, inplace=True)

cellstodrop = counts_per_cell.index[counts_per_cell<(10**countcutoff)]
counts.drop(cellstodrop, axis=0, inplace=True)
TPM.drop(cellstodrop, axis=0, inplace=True)

save_df_to_npz(TPM, './data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM.npz')
save_df_to_npz(counts, './data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_counts.npz')


##
## Get the top 2000 over-dispersed genes
## 

(genestats, gene_fano_parameters) = get_highvar_genes(TPM, numgenes=2000, minimal_mean=0)

axes = make_genestats_plot(genestats, highvarcol='high_var')


genestats_fn = './data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM_genestats.txt'
highvar_genes_fn = './data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_filt_TPM_filter_2000hvgs.txt'
genestats.to_csv(genestats_fn, sep='\t')
highvar_genes = genestats.index[genestats['high_var']]
open(highvar_genes_fn, 'w').write('\n'.join(highvar_genes))
