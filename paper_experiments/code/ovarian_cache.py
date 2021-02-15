
import numpy as np
import pandas as pd
import pickle
import causaldag as cd
import networkx as nx
import itertools as itr
import random 

from R_algs.lrps import lrps, lrps_cv, lrps_path

def abs_cov_mat(X, Y):
    return np.abs(X.T.dot(Y)) / X.shape[0]

# True gene-TF adjacency matrix
amat_true = pd.read_csv("../data/real_data/networkA.csv")
amat_true = amat_true.drop(['Unnamed: 0'], axis=1)
amat_true.index = amat_true.columns 

ALL_OVARIAN_CACHE = dict()

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

# Save gene expression data
gene_data = complete_data.loc[:, ~complete_data.columns.isin(tfs)].copy()
ALL_OVARIAN_CACHE['gene_data'] = gene_data

N_genes = gene_data.shape[1]

data_full = pd.concat([tf_data, gene_data], axis=1)
amat_full = amat_true.loc[data_full.columns, data_full.columns].copy()

ALL_OVARIAN_CACHE['data_full'] = data_full
ALL_OVARIAN_CACHE['amat_full'] = amat_full


tf_amat = amat_full.loc[:, tf_data.columns].copy()
genes_amat =  amat_true.loc[:, gene_data.columns].copy()


########################################################################
######################### Make Evaluation Pairs ########################
########################################################################
samp_abs_cov_mat = abs_cov_mat(tf_data.values, gene_data.values)

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

ALL_OVARIAN_CACHE['node_parent_set_tuples'] = node_parent_set_tuples

random.seed(42)
rand_undir_edges = random.sample(gene_graph.skeleton, 500)

node_pair_del_tuples = []
for edge in rand_undir_edges:
    target_node, del_node = edge
    true_parent_set = sorted(list(gene_graph.neighbors_of(target_node)))
    node_pair_del_tuples.append((target_node, frozenset(true_parent_set)))
    node_pair_del_tuples.append((target_node, frozenset([node for node in true_parent_set if node != del_node])))

ALL_OVARIAN_CACHE['node_pair_del_tuples'] = node_pair_del_tuples

########################################################################
############################# LRPS CV Path #############################
########################################################################

data_obs = gene_data.values.copy()
cov_lrps_path = lrps_path(data_obs)
pickle.dump(cov_lrps_path, open('../results/ovarian_path_lrps_cov_mat.npy', 'wb'))

ALL_OVARIAN_CACHE['cov_lrps_path'] = cov_lrps_path


pickle.dump(ALL_OVARIAN_CACHE, open('../results/ALL_OVARIAN_CACHE.pkl', 'wb'))

