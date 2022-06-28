#!/usr/bin/env python
# coding: utf-8

# ## Example code to illustrate the API for cNMF on simulated data
#
#  - Current as of November 18, 2020
#  - Email dkotliar@broadinstitute.org with questions
#
# ## The code below by default does not use any parallelization but provides example commands for using parallel or a UGER scheduler for running the factorization in parallel

# In[1]:


get_ipython().run_line_magic("matplotlib", "inline")


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import gridspec

# ## Download the example data from the web

# In[3]:


system(
    " !wget -O ./example_simulated_data.tar.gz https://storage.googleapis.com/sabeti-public/dkotliar/cNMF/example_data_20191024.tar.gz"
)
system(
    " !tar -zxvf ./example_simulated_data.tar.gz && rm ./example_simulated_data.tar.gz"
)


# ## cNMF run parameters

# In[4]:


numiter = 20  ## Set this to a larger value for real data. We set this to a low value here for illustration
numworkers = 1  ## Set this to a larger value and use the parallel code cells to try out parallelization
numhvgenes = (
    1500  ## Number of over-dispersed genes to use for running the factorizations
)
K = " ".join([str(i) for i in range(5, 10)])

## Results will be saved to [output_directory]/[run_name] which in this example is simulated_example_data/example_cNMF
output_directory = "./simulated_example_data"
run_name = "example_cNMF"

countfn = "./simulated_example_data/filtered_counts.txt"
seed = 14


# In[5]:


print(K)


# ## Prepare the normalized count matrix of highvar genes and the cNMF parameters file assuming no parallelization
#  - This will normalize the counts matrix and select the 2000 most over-dispersed genes for running cNMF on.
#  - It indicates that it will run 5 NMF iterations each for K=4, 5, 6, 7, and 8. With one worker

# In[6]:


prepare_cmd = (
    "python ../cNMF/cnmf.py prepare --output-dir %s --name %s -c %s -k %s --n-iter %d --total-workers %d --seed %d --numgenes %d"
    % (output_directory, run_name, countfn, K, numiter, numworkers, seed, numhvgenes)
)
print("Prepare command:\n%s" % prepare_cmd)
system(" {prepare_cmd}")


# ## Run the factorization step of cNMF without any parallelization
#
# This might take a few minutes to a half hour depending on how many cores there are on your machine.  You can lower the number of iterations or the K values being considered to speed this up.
#
# As some more explanation, all of the factorization tasks were allocated in the prepare step above to a single worker (worker # 0). We are now executing the factorization steps for that single worker.
#
# In our testing, this took less than a minute to run

# In[7]:


factorize_cmd = (
    "python ../cNMF/cnmf.py factorize --output-dir %s --name %s --worker-index 0"
    % (output_directory, run_name)
)
print("Factorize command for worker 0:\n%s" % factorize_cmd)
get_ipython().system(" {factorize_cmd}")


# ## Run replicates of cNMF for multiple K values in parallel
#
# If more than one total worker is specified, we would need to allocate the tasks for each worker to a distinct processor to work in parallel. There are many ways to do that depending on your compute infastructure.
#
# Here we provide example commands for how you would run the factorization step for cNMF using GNU parallel or UGER if you allocated the tasks amongst 5 workers. First you would need to run the prepare step to specify 5 total workers like below

# In[8]:


numiter = 5  ## Set this to a larger value for real data. We set this to a low value here for illustration
parallelize_numworkers = 5  ## This will parallelize the jobs over 5 workers
numhvgenes = 2000
run_name = "example_cNMF"
K = " ".join([str(i) for i in range(4, 9)])
output_directory = "./simulated_example_data"
countfn = "./simulated_example_data/filtered_counts.txt"
seed = 14

prepare_cmd = (
    "python ../cNMF/cnmf.py prepare --output-dir %s --name %s -c %s -k %s --n-iter %d --total-workers %d --seed %d --numgenes %d"
    % (
        output_directory,
        run_name,
        countfn,
        K,
        numiter,
        parallelize_numworkers,
        seed,
        numhvgenes,
    )
)
print("Prepare command allocating 5 workers:\n%s" % prepare_cmd)
#! {prepare_cmd}


# Then you could use submission commands such as those specified below

# In[9]:


## Using GNU parallel
worker_index = " ".join([str(x) for x in range(parallelize_numworkers)])
cmd = (
    "nohup parallel python ../cNMF/cnmf.py factorize --output-dir %s --name %s --worker-index {} ::: %s"
    % (output_directory, run_name, worker_index)
)
print(cmd)
#!{cmd}


# In[10]:


## Using UGER
logdir = "./log"
cmd = (
    "qsub -cwd -b y -l h_vmem=2g,h_rt=3:00:00 -o %s -e %s -N cnmf -t 1-%d 'python ../cnmf.py factorize --output-dir %s --name %s --worker-index $SGE_TASK_ID'"
    % (logdir, logdir, parallelize_numworkers, output_directory, run_name)
)
print(cmd)
#!{cmd}


# ## Combine the replicate spectra into merged spectra files
#
# Now that the individual factorization replicates have been run, we need to combine them into a single file for each value of K tested

# In[11]:


cmd = "python ../cNMF/cnmf.py combine --output-dir %s --name %s" % (
    output_directory,
    run_name,
)
print("Combine command:\n%s" % cmd)
get_ipython().system("{cmd}")


# ## Plot the trade-off between error and stability as a function of K to guide selection of K
#
# There is no perfect way to choose the value of K for cNMF or for any matrix factorization or clustering algorithm. One approach that can be helpful is to plot the trade-off between solution stability and solution error. We can plot that with the command below

# In[12]:


plot_K_selection_cmd = (
    "python ../cNMF/cnmf.py k_selection_plot --output-dir %s --name %s"
    % (output_directory, run_name)
)
print("Plot K tradeoff command:\n%s" % plot_K_selection_cmd)
get_ipython().system("{plot_K_selection_cmd}")


# ### The plot was just saved to ./simulated_example_data/example_cNMF/example_cNMF.k_selection.pdf

# In[13]:


Image(
    filename="./simulated_example_data/example_cNMF/example_cNMF.k_selection.png",
    width=1000,
    height=1000,
)


# Based on the above plot, we would be interested in investigating Ks around K=7 as a starting point as these are values where the stability has plateaued.

# ## We proceed to obtain the consensus matrix factorization estimates
#
# We first look at how the results look without filtering and then set a threshold for filtering outliers based on the consensus clustergram

# In[14]:


selected_K = 7


# In[15]:


## This is the command you would run from the command line to obtain the consensus estimate with no filtering
## and to save a diagnostic plot as a PDF
consensus_cmd = (
    "python ../cNMF/cnmf.py consensus --output-dir %s --name %s --local-density-threshold %.2f --components %d --show-clustering"
    % (output_directory, run_name, 2.00, selected_K)
)
print("Consensus command for K=%d:\n%s" % (selected_K, consensus_cmd))
get_ipython().system("{consensus_cmd}")


# In[16]:


from IPython.display import Image

Image(
    filename="./simulated_example_data/example_cNMF/example_cNMF.clustering.k_%d.dt_2_00.png"
    % selected_K,
    width=1000,
    height=1000,
)


# This looks reasonable. We are finding 7 clusters as expected. In this case, there aren't any noisey outlier components to filter before clustering. However, in general there will be and we would filter them out by setting a threshold on the local density histogram. For example, you could set a threshold of 0.1 like below

# In[17]:


density_threshold = 0.10
density_threshold_str = "0_10"


# In[18]:


consensus_cmd = (
    "python ../cNMF/cnmf.py consensus --output-dir %s --name %s --local-density-threshold %.2f --components %d --show-clustering"
    % (output_directory, run_name, density_threshold, selected_K)
)
print("Command: %s" % consensus_cmd)
get_ipython().system("{consensus_cmd}")


# In[19]:


from IPython.display import Image

Image(
    filename="./simulated_example_data/example_cNMF/example_cNMF.clustering.k_%d.dt_%s.png"
    % (selected_K, density_threshold_str),
    width=1000,
    height=1000,
)


# In[20]:


get_ipython().system(" ls simulated_example_data/example_cNMF")


# ### Our intended final results files are:
#
#  - example_cNMF.gene_spectra_score.k_6.dt_0_40.txt
#  - example_cNMF.gene_spectra_tpm.k_6.dt_0_40.txt
#  - example_cNMF.usages.k_6.dt_0_40.consensus.txt
#
# The first 2 contain the GEPs re-fit to all of the genes either in units of tpm (example_cNMF.gene_spectra_tpm.k_6.dt_0_40.txt) or in Z-score units (example_cNMF.gene_spectra_score.k_6.dt_0_40.txt). The usage matrix is (example_cNMF.usages.k_6.dt_0_40.consensus.txt)

# ## Now lets make a few plots to illustrate the results

# ### Run PCA on high-var genes TPM normalized
#
# This step can be skipped if you want to just load the pre-generated tsne results in ./simulated_example_data/tsne.txt
from bhtsne import tsne

spectra_hvgs = pd.read_csv(
    "./simulated_example_data/example_cNMF/example_cNMF.spectra.k_%d.dt_%s.consensus.txt"
    % (selected_K, density_threshold_fn),
    sep="\t",
    index_col=0,
)
spectra_hvgs.head()

TPM = load_df_from_npz(
    "./simulated_example_data/example_cNMF/cnmf_tmp/example_cNMF.tpm.df.npz"
)  ## This is the TPM (transcripts per million normalized) file that was saved as an intermediate
TPM.head()

TPM_hvgs = TPM.loc[:, spectra_hvgs.columns]  ## Subset only our list of highvar genes

## And run PCA
PCs = PCA(n_components=10).fit_transform(preprocessing.scale(TPM_hvgs))
PCs = pd.DataFrame(
    PCs, index=TPM_hvgs.index, columns=["PC%d" % i for i in range(1, 11)]
)
PCs.head()

## And run tSNE
tsne_results = pd.DataFrame(
    tsne(PCs.values), index=PCs.index, columns=["TSNE1", "TSNE2"]
)
tsne_results.to_csv("./simulated_example_data/tsne.txt", sep="\t")
# ### Plot the ground truth as well as the cNMF inferences on a tSNE plot

# In[21]:


usage_matrix = pd.read_csv(
    "./simulated_example_data/example_cNMF/example_cNMF.usages.k_%d.dt_%s.consensus.txt"
    % (selected_K, density_threshold_str),
    sep="\t",
    index_col=0,
)
usage_matrix.columns = np.arange(1, selected_K + 1)
normalized_usage_matrix = usage_matrix.div(usage_matrix.sum(axis=1), axis=0)
normalized_usage_matrix.head()


# In[22]:


tsne_results = pd.read_csv("./simulated_example_data/tsne.txt", sep="\t", index_col=0)
tsne_results.head()


# In[23]:


from collections import Counter

import palettable
from matplotlib import rcParams

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Myriad Pro"]

rcParams["axes.titlesize"] = 9
rcParams["axes.labelsize"] = 9
rcParams["xtick.labelsize"] = 7
rcParams["ytick.labelsize"] = 7

rcParams["xtick.major.pad"] = "1"
rcParams["ytick.major.pad"] = "1"

import matplotlib as mpl

label_size = 8

core_colors = type("CoreColors", (), {})

cnames = ["red", "blue", "green", "purple", "orange", "yellow", "brown", "pink", "grey"]


def to_array_col(color):
    return np.array(color) / 255.0


for cname, c in zip(cnames, palettable.colorbrewer.qualitative.Set1_9.colors):
    setattr(core_colors, cname, np.array(c) / 255.0)

for cname, c in zip(
    ["blue", "green", "red", "orange", "purple"],
    palettable.colorbrewer.qualitative.Paired_10.colors[::2],
):
    setattr(core_colors, "pale_" + cname, np.array(c) / 255.0)

core_colors.teal = to_array_col(palettable.colorbrewer.qualitative.Set2_3.colors[0])
core_colors.brown_red = to_array_col(
    palettable.colorbrewer.qualitative.Dark2_3.colors[1]
)

# core_colors.light_grey = to_array_col(palettable.colorbrewer.qualitative.Set2_8.colors[-1])
core_colors.light_grey = to_array_col(palettable.tableau.TableauLight_10.colors[7])

core_colors.royal_green = to_array_col(palettable.wesanderson.Royal1_4.colors[0])


# In[24]:


celldata = pd.read_csv(
    "./simulated_example_data/groundtruth_cellparams.txt", sep="\t", index_col=0
)
celldata.head()


# In[25]:


tsnedat = pd.merge(left=tsne_results, right=celldata, left_index=True, right_index=True)

tsnedat["extra-status"] = "Normal"
tsnedat.loc[tsnedat["has_program"] & ~tsnedat["is_doublet"], "extra-status"] = "program"
tsnedat.loc[~tsnedat["has_program"] & tsnedat["is_doublet"], "extra-status"] = "doublet"
tsnedat.loc[
    tsnedat["has_program"] & tsnedat["is_doublet"], "extra-status"
] = "program & doublet"


# In[26]:


cell_tableau10_color_id = tsnedat["group"].astype(int).values - 1
cell_tableau10_color = (
    np.array(palettable.tableau.Tableau_10.colors[:-1])[cell_tableau10_color_id] / 255.0
)

activity_color = np.zeros((tsnedat.shape[0], 4))
activity_color[:, 3] = tsnedat["has_program"].astype(float)
doublet_color = np.array(palettable.tableau.TableauMedium_10.colors[5]) / 255.0


# In[27]:


normal_filter = tsnedat["extra-status"] == "Normal"
doublet_filter = tsnedat["extra-status"] == "doublet"
activity_filter = tsnedat["extra-status"] == "program"
activity_and_doublet_filter = tsnedat["extra-status"] == "program & doublet"


# In[28]:


identityK = 6
fig = plt.figure(figsize=(1.5, 2), dpi=600)
gs = gridspec.GridSpec(
    3,
    1,
    fig,
    0,
    0,
    1,
    1,
    hspace=0,
    wspace=0,
    height_ratios=[0.2, 1.5, 0.3],
    width_ratios=[1],
)

ax = fig.add_subplot(
    gs[1, 0], xscale="linear", yscale="linear", frameon=False, xticks=[], yticks=[]
)

x = tsnedat["TSNE1"]
y = tsnedat["TSNE2"]


curr_filter = normal_filter


ax.scatter(
    x[curr_filter],
    y[curr_filter],
    facecolor=cell_tableau10_color[curr_filter.values],
    edgecolor="none",
    rasterized=True,
    s=2,
)

curr_filter = activity_filter
ax.scatter(
    x[curr_filter],
    y[curr_filter],
    facecolor=cell_tableau10_color[curr_filter.values],
    edgecolor=activity_color[curr_filter.values],
    linewidth=0.33,
    rasterized=False,
    s=2,
)


curr_filter = doublet_filter | activity_and_doublet_filter
ax.scatter(
    x[curr_filter],
    y[curr_filter],
    c=doublet_color,
    edgecolor="none",
    marker="x",
    linewidth=0.4,
    s=1,
)

curr_filter = activity_and_doublet_filter
ax.scatter(
    x[curr_filter],
    y[curr_filter],
    facecolor="none",
    edgecolor=activity_color[curr_filter.values],
    linewidth=0.33,
    rasterized=False,
    s=2,
)


ax = fig.add_subplot(
    gs[0, 0], frameon=False, xticks=[], yticks=[], xlim=[0, 1], ylim=[0, 1]
)

ax.text(
    0.50,
    0.5,
    "Simulation overview",
    va="center",
    ha="center",
    fontsize=9,
    fontdict=dict(weight="bold"),
    clip_on=False,
)


ax = fig.add_subplot(
    gs[2, 0], frameon=False, xticks=[], yticks=[], xlim=[0, 1], ylim=[0, 1]
)

ax.text(
    0.50,
    0.9,
    "%d cell identity programs" % identityK,
    va="center",
    ha="center",
    fontsize=7,
    clip_on=False,
)

leg_x = np.arange(identityK) / 18
leg_x -= leg_x.mean()
leg_x += 0.5

ax.scatter(
    leg_x,
    np.ones(identityK) * 0.65,
    c=np.array(palettable.tableau.Tableau_10.colors[:identityK]) / 255,
    s=3,
)


activity_y = 0.30
ax.text(
    0.33,
    activity_y,
    "  activity",
    verticalalignment="center",
    ha="center",
    fontsize=7,
    clip_on=False,
)

ax.scatter([0.21], [activity_y], facecolor="none", edgecolor="k", linewidth=0.8, s=4)


doub_y = activity_y
ax.text(
    0.66,
    doub_y,
    "     doublets",
    verticalalignment="center",
    ha="center",
    fontsize=7,
    clip_on=False,
)

ax.scatter(
    [0.54],
    [doub_y],
    c=[doublet_color],
    edgecolor="none",
    marker="x",
    linewidth=0.8,
    s=4,
    clip_on=False,
)

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])


# In[29]:


tsne_and_usages = pd.merge(
    left=normalized_usage_matrix, right=tsne_results, left_index=True, right_index=True
)
tsne_and_usages.head()


# In[30]:


cmap = palettable.cartocolors.sequential.Sunset_7.get_mpl_colormap()

(fig, axes) = plt.subplots(3, 3, figsize=(5, 4), dpi=400)
axes = axes.ravel()
for i in range(1, selected_K + 1):
    sc = axes[i - 1].scatter(
        tsne_and_usages["TSNE1"],
        tsne_and_usages["TSNE2"],
        c=tsne_and_usages[i],
        cmap=cmap,
        vmin=0,
        vmax=1.0,
        s=2,
        edgecolor="none",
        rasterized=True,
    )
    axes[i - 1].set_title(i)
    axes[i - 1].set_xticks([])
    axes[i - 1].set_yticks([])


plt.tight_layout()
cbarax = fig.add_axes([1, 0.7, 0.02, 0.2])
plt.colorbar(sc, cax=cbarax)
_ = plt.figtext(0.45, 1, "GEP Usage", fontsize=15)
axes[-1].remove()
axes[-2].remove()


# The plot above shows which cells are expressing which programs. The first program is distributed over cells of multiple identity programs and therefore corresponds to the activity program. GEP 2-6 correspond to distinct identity programs

# We might also want to investigate which genes are driving which programs. We can do so by loading the Z-score transformed spectra and sorting it as follows

# In[31]:


gene_scores = pd.read_csv(
    "./simulated_example_data/example_cNMF/example_cNMF.gene_spectra_score.k_%d.dt_%s.txt"
    % (selected_K, density_threshold_str),
    sep="\t",
    index_col=0,
)
gene_scores = gene_scores.T
gene_scores.head()


# ## Below we plot the top 10 genes associated with each gene expression program

# In[32]:


topgenes = []
num_top_genes = 10
for gep in gene_scores.columns:
    topgenes.append(
        pd.Series(gene_scores[gep].sort_values(ascending=False).index[:num_top_genes])
    )

topgenes = pd.concat(topgenes, axis=1)
topgenes.columns = ["GEP%d" % g for g in gene_scores.columns]


# In[33]:


topgenes


# In[ ]:
