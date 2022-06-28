#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:14:11 2021

@author: asif
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from bhtsne import tsne
import glob, os
sys.path.append('../cNMF')
from cnmf import get_highvar_genes, save_df_to_npz, load_df_from_npz
from plotting import make_genestats_plot


from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Myriad Pro'
rcParams['font.sans-serif'] = ['Myriad Pro']
rcParams['axes.titlesize'] = 9
rcParams['axes.labelsize'] = 9
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7


"""
cluster_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71585/suppl/GSE71585_Clustering_Results.csv.gz'
cluster_fn = '../../../data/Part3_VisualCortex/GSE71585_Clustering_Results.csv.gz'

rpkm_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71585/suppl/GSE71585_RefSeq_RPKM.csv.gz'
rpkm_fn = '../../../data/Part3_VisualCortex/GSE71585_RefSeq_RPKM.csv.gz'

count_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71585/suppl/GSE71585_RefSeq_counts.csv.gz'
count_fn = '../../../data/Part3_VisualCortex/GSE71585_RefSeq_counts.csv.gz'

tpm_url = 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE71nnn/GSE71585/suppl/GSE71585_RefSeq_TPM.csv.gz'
tpm_fn = '../../../data/Part3_VisualCortex/GSE71585_RefSeq_TPM.csv.gz'

if not os.path.exists('../../../data/Part3_VisualCortex'):
    os.mkdir('../../../data/Part3_VisualCortex')

cmd = 'wget -O %s %s; gzip -f -d %s'% (cluster_fn, cluster_url, cluster_fn)
print(cmd)
!{cmd}

cmd = 'wget -O %s %s; gzip -f -d %s'% (rpkm_fn, rpkm_url, rpkm_fn)
print(cmd)
!{cmd}

cmd = 'wget -O %s %s; gzip -f -d %s'% (count_fn, count_url, count_fn)
print(cmd)
!{cmd}

cmd = 'wget -O %s %s; gzip -f -d %s'% (tpm_fn, tpm_url, tpm_fn)
print(cmd)
!{cmd}
"""

##
## Convert to counts per kb of trasncript
##

rpkm = pd.read_csv('./data/Part3_VisualCortex/GSE71585_RefSeq_RPKM.csv', index_col=0).T
counts = pd.read_csv('./data/Part3_VisualCortex/GSE71585_RefSeq_counts.csv', index_col=0).T

counts_per_kb = rpkm.multiply(counts.sum(axis=1)/(10**6), axis=0)

save_df_to_npz(counts_per_kb, './data/Part3_VisualCortex/GSE71585_RefSeq_CountsPerKb.npz')


##
## Keep only the neurons (defined based on the previous clustering) and exclude the other cell-types
##

counts_per_kb = load_df_from_npz('./data/Part3_VisualCortex/GSE71585_RefSeq_CountsPerKb.npz')

celltype = pd.read_csv('./data/Part3_VisualCortex/GSE71585_Clustering_Results.csv', index_col=0)
celltype.head()

celltype['broad_type'].value_counts()

neuron_idx = celltype.index[celltype['broad_type'].isin(['Glutamatergic Neuron', 'GABA-ergic Neuron'])]
counts_per_kb = counts_per_kb.loc[neuron_idx,:]
print('%d cells by %d genes remaining' % counts_per_kb.shape)


##
## Drop outlier cells that were identified by PCA and were frequently getting their own component by cNMF
##

outliers = ['Nos1_tdTpositive_cell_72', 'PvalbF-Gad2_tdTpositive_cell_66', 'Vip_tdTpositive_cell_29',
           'Vip_tdTpositive_cell_61', 'Cux2_tdTpositive_cell_35', 'Rorb_tdTpositive_cell_13',
           'Calb2_tdTpositive_cell_62']

counts_per_kb = counts_per_kb.drop(outliers, axis=0)


##
## Drop genes that were not in the Hrvatin Et Al dataset as well
## 

hrvatinetal_genes = load_df_from_npz('./data/Part3_VisualCortex/GSE102827_merged_ExcitatoryInhibatoryNeurons_raw.npz')
hrvatinetal_genes = hrvatinetal_genes.columns

summary = (counts_per_kb.shape[1], len(hrvatinetal_genes),   len(set(counts_per_kb.columns).intersection(set(hrvatinetal_genes))))
'%d genes in Tasic Et. Al, %d genes in Hrvatin Et Al, %d genes in intersection' % summary

overlap_genes = sorted(set(counts_per_kb.columns).intersection(set(hrvatinetal_genes)))

counts_per_kb = counts_per_kb.loc[:,overlap_genes]


##
## Filter cells that have lost too large a fraction of their counts in the previous filters
##

TPM = pd.read_csv('./data/Part3_VisualCortex/GSE71585_RefSeq_TPM.csv', index_col=0).T

(fig,ax) = plt.subplots(1,1, figsize=(3,2), dpi=400)
remaining_tpm_cutoff = 785000
remaining_TPM = TPM.loc[counts_per_kb.index, counts_per_kb.columns].sum(axis=1)
_ = ax.hist(remaining_TPM, bins=100)
ax.vlines(x=remaining_tpm_cutoff,ymin=0,ymax=150, linestyle='--')
ax.set_ylim([0,150])
ax.set_xlabel('Remaining TPM After Filters')
ax.set_ylabel('Number of Cells')

cellstodrop = remaining_TPM.index[remaining_TPM<=remaining_tpm_cutoff]
counts_per_kb.drop(cellstodrop, axis=0, inplace=True)

##
## Filter cells that are outliers in terms of their distance to their K nearest neighbors
##

import pysparnn.cluster_index as ci
from scipy.sparse import csr_matrix

cells = list(counts_per_kb.index)
genes = list(counts_per_kb.columns)
counts_per_kb_sparse = csr_matrix(counts_per_kb.values)
K=11

data_to_return = range(counts_per_kb_sparse.shape[0])
cp = ci.MultiClusterIndex(counts_per_kb_sparse, data_to_return)
res = cp.search(counts_per_kb_sparse, k=K, return_distance=True)
distance_to_10nn = np.array([[sampdat[0] for sampdat in row] for row in res])
neighbor_cells = np.array([[cells[int(sampdat[1])] for sampdat in row] for row in res])
distance_to_10nn = pd.DataFrame(distance_to_10nn, index=cells).iloc[:,1:]
neighbor_cells = pd.DataFrame(neighbor_cells, index=cells).iloc[:,1:]
meandist_to_10nn = distance_to_10nn.mean(axis=1)

cutoff = .12
(fig,ax) = plt.subplots(1,1, figsize=(3,2), dpi=400)
_ = ax.hist(meandist_to_10nn, bins=100)
ax.vlines(x=cutoff, ymin=0, ymax=220, linestyle='--')
ax.set_xlabel('Mean Distance to 10 Nearest Neighbors')
ax.set_ylabel('Number of Cells')
ax.set_ylim([0,220])

cellstodrop = meandist_to_10nn.index[meandist_to_10nn>cutoff]
len(cellstodrop), len(meandist_to_10nn)

counts_per_kb.drop(cellstodrop, axis=0, inplace=True)

##
## Convert to TPM
##

TPM = counts_per_kb.div(counts_per_kb.sum(axis=1), axis=0) * (10**6)

##
## Filter genes that are detected in too few cells
##

nnzthresh = counts_per_kb.shape[0] / 100
numnonzero = (counts_per_kb>0).sum(axis=0)
print((numnonzero>nnzthresh).value_counts())
ind = numnonzero<200
(fig,ax) = plt.subplots(1,1, figsize=(3,2), dpi=400)
_ = ax.hist(numnonzero.loc[ind], bins=100)
(_,ymax) = ax.get_ylim()
ax.vlines(x=nnzthresh, ymin=0, ymax=ymax, linestyle='--')
ax.set_xlabel('# Samples With Non-zero Count')
ax.set_ylabel('# Genes')
              
(fig,ax) = plt.subplots(1,1, figsize=(2,2), dpi=300)
logexpcutoff=-2
genemean = TPM.mean(axis=0)
todrop = genemean.index[genemean==0]
counts_per_kb.drop(todrop, axis=1, inplace=True)
TPM.drop(todrop, axis=1, inplace=True)
numnonzero.drop(todrop, inplace=True)
genemean = genemean.loc[genemean>0]
ax.hist(genemean.loc[numnonzero>nnzthresh].apply(np.log10), bins=100, color='r', label='Pass thresh', alpha=0.7)
ax.hist(genemean.loc[numnonzero<=nnzthresh].apply(np.log10), bins=100, color='b', label='Fail thresh', alpha=0.7)
_ = ax.set_title('Log10 Mean Gene Expression')
lims = ax.get_ylim()
ax.vlines(x=logexpcutoff, ymin=0, ymax=lims[1], linestyle='--')
ax.legend(fontsize=4)
_ = ax.set_ylim(lims)

todrop = genemean.index[(genemean.apply(np.log10)<logexpcutoff) | (numnonzero<=nnzthresh)]
counts_per_kb.drop(todrop, axis=1, inplace=True)
TPM.drop(todrop, axis=1, inplace=True)

save_df_to_npz(TPM, './data/Part3_VisualCortex/GSE71585_RefSeq_TPM.filtered.npz')
save_df_to_npz(counts_per_kb, './data/Part3_VisualCortex/GSE71585_RefSeq_CountsPerKb.filtered.npz')


##
## Get high-variance genes
##

(genestats, gene_fano_parameters) = get_highvar_genes(TPM, numgenes=2000, minimal_mean=0) 

axes = make_genestats_plot(genestats, highvarcol='high_var')

genestats_fn = './data/Part3_VisualCortex/GSE71585_RefSeq_filt_TPM_genestats.txt'
highvar_genes_fn = './data/Part3_VisualCortex/GSE71585_RefSeq_filt_TPM_2000hvgs.txt'
genestats.to_csv(genestats_fn, sep='\t')
highvar_genes = genestats.index[genestats['high_var']]
open(highvar_genes_fn, 'w').write('\n'.join(highvar_genes))

