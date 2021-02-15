
import sys
sys.path.append("../code/")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import causaldag as cd
import networkx as nx
import itertools as itr
import random 
import pickle

from decamfound.pcss import *
from utils.metrics import compute_true_prob

import os
from time import time
import numpy as np
from models.baseline_models import *
from models.all_models import *

import seaborn as sns
sns.set(style='white',)

def abs_cov_mat(X, Y, make_abs=True):
    if make_abs:
        return np.abs(X.T.dot(Y)) / X.shape[0]
    else:
        return X.T.dot(Y) / X.shape[0]


########################################################################
######################### Exploratory Analysis #########################
########################################################################

# True gene-TF adjacency matrix
amat_true = pd.read_csv("../data/real_data/networkA.csv")
amat_true = amat_true.drop(['Unnamed: 0'], axis=1)
amat_true.index = amat_true.columns 

# Data with all genes and TFs
complete_data = pd.read_csv('../data/real_data/cancer_ovarian.csv')

# Make first columns of the dataframe / adjacency matrix 
# correspond to TFs
tfs = sorted(["FOS", "FOSB", "JUN", "JUNB", "JUND", 
       "ESR1", "ESR2", "AR", "NFKB1", "NFKB2", 
       "RELA", "RELB", "REL", "STAT1", "STAT2", 
       "STAT3", "STAT4", "STAT5", "STAT6"])

tf_data = complete_data.loc[:, complete_data.columns.isin(tfs)].copy()
N_tfs = tf_data.shape[1]

tfs = np.array(list(tf_data.columns.copy()))

gene_data = complete_data.loc[:, ~complete_data.columns.isin(tfs)].copy()
N_genes = gene_data.shape[1]

data_full = pd.concat([tf_data, gene_data], axis=1)
amat_full = amat_true.loc[data_full.columns, data_full.columns].copy()

tf_amat = amat_full.loc[:, tf_data.columns].copy()
genes_amat =  amat_true.loc[:, gene_data.columns].copy()

tf_gene_counts = tf_amat.values[N_tfs:, :].sum(axis=0)

pd.DataFrame({'TF': tfs, '# Gene Edges': tf_gene_counts}).to_csv('../data/real_data/tf_gene_counts.csv', index=False)


# Look at linear correlations between TFs and genes
samp_abs_cov_mat = abs_cov_mat(tf_data.values, gene_data.values)
samp_corr_mat = abs_cov_mat(tf_data.values, gene_data.values, False)

print(samp_corr_mat.min()) # smallest correlation is -.423


tf_correlation_df = pd.DataFrame(samp_abs_cov_mat.T.copy(), columns=tf_data.columns)
tf_correlation_df = tf_correlation_df.melt()
tf_correlation_df.columns = ['TF', '|Correlation|']

high_degree_tfs = list(tf_data.columns[tf_gene_counts > 50]) # Edge w/ more than 10% of the genes

high_degree_mask = tf_correlation_df['TF'].isin(high_degree_tfs).values

sns.boxplot(x='TF', y='|Correlation|', data=tf_correlation_df[high_degree_mask],  showfliers = True)
plt.xlabel('Transcription Factor', fontsize=14)
plt.ylabel('Gene Correlation Magnitudes', fontsize=14)
plt.tight_layout()
plt.savefig('../figures/ovarian/tf_correlations.png')
plt.close()

tf_gene_ranks = samp_abs_cov_mat.argsort(axis=1)
sorted_tfs = tf_data.columns[tf_gene_counts.argsort()[::-1]]
pcss_suff_stats, est_factor_component, K, evecs, evals = get_confounding_suff_stats(gene_data.values, K=7)

# Plot the spectrum of the gene data
plt.figure(figsize=(6, 4), dpi=200)
sns.scatterplot(range(evals.shape[0]), evals)
sns.lineplot(range(evals.shape[0]), evals, linestyle='--')
plt.xlabel('Principal Component', fontsize=16)
plt.ylabel('Eigenvalue', fontsize=16)
plt.tight_layout()
plt.savefig('../figures/ovarian/scree_plot.png')
plt.close()


def make_tf_scatter_plot(tf_data, gene_data, gene_name, tf_name, ax, **kwargs):
    tf_ix = np.where(tf_data.columns == tf_name)[0][0]
    gene_ix = np.where(gene_data.columns == gene_name)[0][0]
    sns.scatterplot(tf_data.values[:, tf_ix], 
                    gene_data.values[:, gene_ix], ax=ax, **kwargs)

    ax.set_xlabel('TF: {0}'.format(tf_name))
    ax.set_ylabel('Gene: {0}'.format(gene_name))
    return tf_ix, gene_ix


def make_tf_pcss_plot(tf_data, gene_data, pcss, tf_names, n_top, savepath):
    samp_abs_cov_mat = abs_cov_mat(tf_data.values, gene_data.values)
    tf_gene_ranks = samp_abs_cov_mat.argsort(axis=1)
    all_gene_names = list(gene_data.columns)
    fig, ax = plt.subplots(n_top, len(tf_names), figsize=(10, 10), dpi=200)
    for n_tf, tf_name in enumerate(tf_names):
        for rank in range(n_top):
            tf_ix = np.where(tf_data.columns == tf_name)[0][0]
            gene_name = all_gene_names[tf_gene_ranks[tf_ix, -rank - 1]]
            __, gene_ix = make_tf_scatter_plot(tf_data, gene_data, gene_name, 
                                 tf_name, ax[rank, n_tf], 
                                 color='blue', marker='x', 
                                 alpha=.5, label='data')

            sns.scatterplot(tf_data.values[:, tf_ix], 
                            pcss[:, gene_ix], ax=ax[rank, n_tf], 
                            color='green', alpha=.5, label='pcss')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


make_tf_pcss_plot(tf_data, gene_data, pcss_suff_stats, 
                 ['NFKB1', 'FOS', 'STAT1'], 3, 
                '../figures/ovarian/tf_vs_pcss.png')


# Make NKB1 Plot
nfkb1_ix = np.where(tf_data.columns == 'NFKB1')[0][0]
top_nfk1_gene_ix = tf_gene_ranks[nfkb1_ix, -1]

sns.scatterplot(tf_data.values[:, nfkb1_ix], 
                    gene_data.values[:, top_nfk1_gene_ix], label='BIRC3', marker='x')

sns.scatterplot(tf_data.values[:, nfkb1_ix], 
                    pcss_suff_stats[:, top_nfk1_gene_ix], label='BIRC3-pcss')

plt.ylabel('Expression Level', fontsize=14)
plt.xlabel('Latent TF NFKB1', fontsize=14)
plt.ylim(-3, 3)
plt.tight_layout()
plt.tight_layout()
plt.savefig('../figures/ovarian/NFKB1_BIRC3.png')
plt.close()

smallest_cor_tf = samp_abs_cov_mat[:, top_nfk1_gene_ix].argmin()

sns.scatterplot(tf_data.values[:, smallest_cor_tf], 
                    gene_data.values[:, top_nfk1_gene_ix], label='BIRC3', marker='x')

sns.scatterplot(tf_data.values[:, smallest_cor_tf], 
                    pcss_suff_stats[:, top_nfk1_gene_ix], label='BIRC3-pcss')


plt.ylabel('Expression Level', fontsize=14)
plt.xlabel('Latent TF JUN', fontsize=14)
plt.ylim(-3, 3)
plt.tight_layout()
plt.savefig('../figures/ovarian/JUN_BIRC3.png')
plt.close()


nfkb1_neg_gene_ix = samp_corr_mat.argsort(axis=1)[nfkb1_ix, 0]

sns.scatterplot(tf_data.values[:, nfkb1_ix], 
                    gene_data.values[:, nfkb1_neg_gene_ix], marker='x', label='SMARCE1')

sns.scatterplot(tf_data.values[:, nfkb1_ix], 
                    pcss_suff_stats[:, nfkb1_neg_gene_ix], label='SMARCE1-pcss')

plt.ylabel('Expression Level', fontsize=14)
plt.xlabel('Latent TF NFKB1', fontsize=14)
plt.tight_layout()
plt.tight_layout()
plt.savefig('../figures/ovarian/NFKB1_SMARCE1.png')
plt.close()

# Get set of gene pairs that are conditionally independent given
# the TFs.

# Make undirected graph of TFs and genes (TFs) are first nodes
full_undir_graph = cd.UndirectedGraph.from_amat(amat_full.values)

nx_full_undir_graph = nx.convert_matrix.from_numpy_matrix(amat_full.values)

gene_graph = cd.UndirectedGraph.from_amat(amat_full.values[15:, 15:])

nx_gene_undir_graph = nx.convert_matrix.from_numpy_matrix(amat_full.values[15:, 15:])


# All genes have a TF interaction...
np.all((amat_full.values[:N_tfs, N_tfs:] > 0).sum(axis=0) > 0)

# Get set of nodes with shared TF but no edge between them
missing_gene_edges = {frozenset({i, j}) for i, j in itr.combinations(gene_graph.nodes, 2)} - gene_graph.skeleton

genes_with_latent_confounding = set()
for node1, node2 in missing_gene_edges:
    node1_ix = node1 + N_tfs
    node2_ix = node2 + N_tfs
    if amat_full.values[:15, node1_ix].dot(amat_full.values[:15, node2_ix]) > 0: # Genes share a TF
        genes_with_latent_confounding.add(frozenset({node1, node2}))

genes_cov_mat = abs_cov_mat(gene_data.values, gene_data.values)
genes_with_latent_confounding = list(genes_with_latent_confounding)
genes_with_latent_confounding_corrs = np.array([genes_cov_mat[i, j] for i,j in genes_with_latent_confounding])

genes_with_latent_confounding = np.array(genes_with_latent_confounding)[genes_with_latent_confounding_corrs.argsort()[::-1]]
genes_with_latent_confounding_corrs = genes_with_latent_confounding_corrs[genes_with_latent_confounding_corrs.argsort()[::-1]]

top_confound_gene1, top_confound_gene2 = genes_with_latent_confounding[0] # equals genes 92, 316


nx.has_path(nx_gene_undir_graph, top_confound_gene1, top_confound_gene2) # still a path given TFs


amat_full.values[:15, top_confound_gene1 + N_tfs]
amat_full.values[:15, top_confound_gene2 + N_tfs]

shared_tfs_mask = (amat_full.values[:15, top_confound_gene1 + N_tfs] == 1) & (amat_full.values[:15, top_confound_gene2 + N_tfs] == 1)


samp_abs_cov_mat[shared_tfs_mask, top_confound_gene1]
samp_abs_cov_mat[shared_tfs_mask, top_confound_gene2]

amat_full.values[:, top_confound_gene1 + N_tfs].dot(amat_full.values[:, top_confound_gene2 + N_tfs])

# Genes 92 and 416 aren't connected but have the highest correlation
fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=200)

sns.scatterplot(gene_data.values[:, 92], gene_data.values[:, 316], ax=ax[0, 0], color='red')
ax[0, 0].set_xlabel('Gene: {0}'.format(gene_data.columns[92]))
ax[0, 0].set_ylabel('Gene: {0}'.format(gene_data.columns[316]))
ax[0, 0].set_ylim(-3, 3)

# Both genes 92 and 416 have exactly one shared TF, TF 4 = NFKB1
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92], ax=ax[0, 1], label=gene_data.columns[92], alpha=.8)
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316], ax=ax[0, 1], label=(gene_data.columns[316]), alpha=.8)
ax[0, 1].set_xlabel('TF: NFKB1')
ax[0, 1].set_ylabel('Gene Expression')
ax[0, 1].set_ylim(-3, 3)

sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92] - pcss_suff_stats[:, 92], 
                label=gene_data.columns[92], ax=ax[1, 0], alpha=.8)

sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316] - pcss_suff_stats[:, 316], 
                label=gene_data.columns[316], ax=ax[1, 0], alpha=.8)

ax[1, 0].set_xlabel('TF: NFKB1')
ax[1, 0].set_ylabel('Gene Expression - PCSS')

sns.scatterplot(gene_data.values[:, 92] - pcss_suff_stats[:, 92], 
    gene_data.values[:, 316] - pcss_suff_stats[:, 316], ax=ax[1, 1], color='green')

ax[1, 0].set_ylim(-3, 3)

ax[1, 1].set_xlabel('Gene {0} - PCSS'.format(gene_data.columns[92]))
ax[1, 1].set_ylabel('Gene {0} - PCSS'.format(gene_data.columns[316]))

ax[1, 1].set_ylim(-3, 3)

plt.tight_layout()

plt.savefig('../figures/ovarian/before_after_pcss.png')
plt.close()


np.corrcoef(tf_data.values[:, 4], gene_data.values[:, 92]) # .49
np.corrcoef(tf_data.values[:, 4], gene_data.values[:, 316]) # .47

np.corrcoef(tf_data.values[:, 4], gene_data.values[:, 92] - pcss_suff_stats[:, 92]) # -.096
np.corrcoef(tf_data.values[:, 4], gene_data.values[:, 316] - pcss_suff_stats[:, 316]) # -.096



sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92], label=gene_data.columns[92], alpha=.8)
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316], label=(gene_data.columns[316]), alpha=.8)
plt.xlabel('Latent TF NFKB1', fontsize=14)
plt.ylabel('Expression Level', fontsize=14)
plt.legend(fontsize=14)
plt.ylim(-3, 3)
plt.tight_layout()
plt.savefig('../figures/ovarian/confound_tf_gene_pairs.png')
plt.close()


sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92] - pcss_suff_stats[:, 92], label=gene_data.columns[92], alpha=.8)
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316] - pcss_suff_stats[:, 316], label=(gene_data.columns[316]), alpha=.8)
plt.xlabel('Latent TF NFKB1', fontsize=14)
plt.ylabel('Deconfounded Expression Level', fontsize=14)
plt.legend(fontsize=14)
plt.ylim(-3, 3)
plt.tight_layout()
plt.savefig('../figures/ovarian/deconfound_tf_gene_pairs.png')
plt.close()

