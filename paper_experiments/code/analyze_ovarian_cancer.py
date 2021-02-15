
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

def abs_cov_mat(X, Y):
    return np.abs(X.T.dot(Y)) / X.shape[0]

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

gene_data = complete_data.loc[:, ~complete_data.columns.isin(tfs)].copy()
N_genes = gene_data.shape[1]

data_full = pd.concat([tf_data, gene_data], axis=1)
amat_full = amat_true.loc[data_full.columns, data_full.columns].copy()

tf_amat = amat_full.loc[:, tf_data.columns].copy()
genes_amat =  amat_true.loc[:, gene_data.columns].copy()

tf_gene_counts = tf_amat.values[N_tfs:, :].sum(axis=0)

# Look at linear correlations between TFs and genes
samp_abs_cov_mat = abs_cov_mat(tf_data.values, gene_data.values)

tf_correlation_df = pd.DataFrame(samp_abs_cov_mat.T.copy(), columns=tf_data.columns)
tf_correlation_df = tf_correlation_df.melt()
tf_correlation_df.columns = ['TF', '|Correlation|']

high_degree_tfs = list(tf_data.columns[tf_gene_counts > 50]) # Edge w/ more than 10% of the genes

high_degree_mask = tf_correlation_df['TF'].isin(high_degree_tfs).values

sns.boxplot(x='TF', y='|Correlation|', data=tf_correlation_df[high_degree_mask],  showfliers = False)
sns.stripplot(x='TF', y='|Correlation|', data=tf_correlation_df[high_degree_mask & (tf_correlation_df['|Correlation|'] > .4).values], alpha=.5)
plt.xlabel('Transcription Factor', fontsize=14)
plt.ylabel('Gene Correlation Magnitudes', fontsize=14)
plt.tight_layout()
# plt.savefig('../figures/ovarian/tf_correlations.png')
plt.close()

tf_gene_ranks = samp_abs_cov_mat.argsort(axis=1)
sorted_tfs = tf_data.columns[tf_gene_counts.argsort()[::-1]]
pcss_suff_stats, est_factor_component, K, evecs, evals = get_confounding_suff_stats(gene_data.values, K=7)

# Plot the spectrum of the gene data
sns.scatterplot(range(evals.shape[0]), evals)
sns.lineplot(range(evals.shape[0]), evals, linestyle='--')
plt.xlabel('Principal Component', fontsize=14)
plt.ylabel('Eigenvalue', fontsize=14)
plt.tight_layout()
# plt.savefig('../figures/ovarian/scree_plot.png')
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
    # plt.savefig(savepath)
    plt.close()


make_tf_pcss_plot(tf_data, gene_data, pcss_suff_stats, 
                 ['NFKB1', 'FOS', 'STAT1'], 3, 
                '../figures/ovarian/tf_vs_pcss.png')

# Get set of gene pairs that are conditionally independent given
# the TFs.

# Make undirected graph of TFs and genes (TFs) are first nodes
full_undir_graph = cd.UndirectedGraph.from_amat(amat_full.values)
gene_graph = cd.UndirectedGraph.from_amat(amat_full.values[15:, 15:])

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

amat_full.values[:15, 92 + N_tfs]
amat_full.values[:15, 316 + N_tfs]

# Genes 92 and 416 aren't connected but have the highest correlation
fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=200)

sns.scatterplot(gene_data.values[:, 92], gene_data.values[:, 316], ax=ax[0, 0], color='red')
ax[0, 0].set_xlabel('Gene: {0}'.format(gene_data.columns[92]))
ax[0, 0].set_ylabel('Gene: {0}'.format(gene_data.columns[316]))

# Both genes 92 and 416 have exactly one shared TF, TF 4
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92], ax=ax[0, 1], label=gene_data.columns[92], alpha=.8)
sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316], ax=ax[0, 1], label=(gene_data.columns[316]), alpha=.8)
ax[0, 1].set_xlabel('TF: {0}'.format(gene_data.columns[92]))
ax[0, 1].set_ylabel('Gene Expression')

sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 92] - pcss_suff_stats[:, 92], 
                label=gene_data.columns[92], ax=ax[1, 0], alpha=.8)

sns.scatterplot(tf_data.values[:, 4], gene_data.values[:, 316] - pcss_suff_stats[:, 316], 
                label=gene_data.columns[316], ax=ax[1, 0], alpha=.8)

ax[1, 0].set_xlabel('TF: {0}'.format(gene_data.columns[92]))
ax[1, 0].set_ylabel('Gene Expression - PCSS')

sns.scatterplot(gene_data.values[:, 92] - pcss_suff_stats[:, 92], 
    gene_data.values[:, 316] - pcss_suff_stats[:, 316], ax=ax[1, 1], color='green')

ax[1, 1].set_xlabel('Gene {0} - PCSS'.format(gene_data.columns[92]))
ax[1, 1].set_ylabel('Gene {0} - PCSS'.format(gene_data.columns[316]))

plt.tight_layout()

# plt.savefig('../figures/ovarian/before_after_pcss.png')
plt.close()


########################################################################
################# Performance for Parent Set Recovery ##################
########################################################################

strong_tf_shared_edges = set()
for edge in genes_with_latent_confounding:
    gene1, gene2 = edge
    gene1_mask = np.array(samp_abs_cov_mat[:, gene1] > .4, dtype=np.int32)
    gene2_mask =  np.array(samp_abs_cov_mat[:, gene2] > .4, dtype=np.int32)
    if gene1_mask.dot(gene2_mask) > 0:
        strong_tf_shared_edges.add((gene1, gene2))
        strong_tf_shared_edges.add((gene2, gene1)) # Include both edge directions

true_parent_sets = []
wrong_parent_sets = []
node_parent_set_tuples = []
for target_node, wrong_node in strong_tf_shared_edges:
    true_parent_set = sorted(list(gene_graph.neighbors_of(target_node)))
    wrong_parent_sets.append(true_parent_set + [wrong_node])
    true_parent_sets.append(true_parent_set)
    node_parent_set_tuples.append((target_node, frozenset(true_parent_set)))
    node_parent_set_tuples.append((target_node, frozenset(true_parent_set + [wrong_node])))


# Compute covariance matrix from Vanilla, PCSS, & LRPS
data_obs = gene_data.values.copy()
cov_obs = np.cov(data_obs, rowvar=False)
bic_suffstat = dict(cov=cov_obs, nsamples=data_obs.shape[0], samples=data_obs)

# prec_lrps = lrps_cv(data_obs)
# pickle.dump(prec_lrps, open('../results/ovarian_lrps_prec_mat.npy', 'wb'))

vanilla_results = compute_score_parent_set_dag_helper(bic_suffstat, node_parent_set_tuples, method='normal', progress=False)
poet_results = compute_score_parent_set_dag_helper(bic_suffstat, node_parent_set_tuples, method='poet', K=7, progress=False)

prec_lrps = pickle.load(open('../results/real/ovarian_lrps_prec_mat.npy', 'rb'))
bic_suffstat_lrps = dict(cov=np.linalg.inv(prec_lrps), nsamples=data_obs.shape[0], samples=data_obs)
lrps_results = compute_score_parent_set_dag_helper(bic_suffstat_lrps, node_parent_set_tuples, method='normal', progress=False)


gp_suffstats = dict(obs_data=data_obs, true_confounders=data_full.values[:, :N_tfs], pcss=pcss_suff_stats)
decam_results = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_parent_set_tuples, method='decamfounder')
cam_cheat_results = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_parent_set_tuples, method='CAM-CHEAT')
cam_results = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_parent_set_tuples, method='CAM')

poet_lrps_diffs = []
for i in range(0, len(node_parent_set_tuples), 2):
    poet_lrps_diffs.append((lrps__inv_results[node_parent_set_tuples[i]] - poet_results[node_parent_set_tuples[i]]))


all_results = dict()
all_results['vanilla'] = vanilla_results
all_results['poet'] = poet_results
all_results['lrps'] = lrps_results
all_results['cam'] = cam_results
all_results['cam_cheat'] = cam_cheat_results
all_results['decamfound'] = decam_results
all_results['node_parent_set_tuples'] = node_parent_set_tuples

# pickle.dump(all_results, open('../results/real/ovarian_parent_scores.pkl', 'wb'))
