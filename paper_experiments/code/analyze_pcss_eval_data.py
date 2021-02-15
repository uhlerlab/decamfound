
import numpy as np
import causaldag as cd

from decamfound.pcss import get_confounding_suff_stats
from utils.generate_synthetic_data import make_confound_data
from causaldag import DAG, CamDAG
from utils.basis_fns import *

import pickle
import matplotlib.pyplot as plt


def compute_mse_non_linear(confound_dag, data_full, pcss_suff_stats, num_confounders):
	data_obs = data_full[:, num_confounders:]
	confounder_nodes = set(range(num_confounders))
	children_of_confounders = set()
	for confounder in confounder_nodes:
		children_of_confounders = children_of_confounders | confound_dag.children_of(confounder)
	
	bad_children = set()	
	for child in children_of_confounders:
		if len(confound_dag.parents_of(child) - confounder_nodes) > 0:
			bad_children = bad_children | {child}

	children_of_confounders = children_of_confounders - bad_children
	mse_errors = []
	for child in children_of_confounders:
		# Compute true E[x | h]
		child_mean_fn = confound_dag.mean_functions[child]
		actual = np.ones((data_obs.shape[0],)) * child_mean_fn.keywords['intercept']
		for confound_node in child_mean_fn.keywords['confounder_nodes']:
			net_weight = child_mean_fn.keywords['c_node_confound'] 
			net_weight *= child_mean_fn.keywords['parent_weights_confound_dict'][confound_node]
			fn = child_mean_fn.keywords['parent2base_confound'][confound_node]
			confound_fn = lambda x: net_weight * fn(x)
			actual += np.array([confound_fn(x) for x in data_full[:, confound_node]])

		pcss_est = pcss_suff_stats[:, child - num_confounders] # Need to reindex
		mse_errors.append(np.mean((actual - pcss_est) ** 2))

	return np.array(mse_errors)


def compute_mse_linear(confound_dag, data_full, pcss_suff_stats, num_confounders):
	data_obs = data_full[:, num_confounders:]
	confounder_nodes = set(range(num_confounders))
	p_full = data_full.shape[1]
	p = p_full - num_confounders
	B_full = np.zeros((p_full, p_full))
	intercepts = np.zeros(p_full)

	for node in range(num_confounders, p_full):
		node_mean_fn = confound_dag.mean_functions[node]
		for confound_node in node_mean_fn.keywords['parent_weights_confound_dict'].keys():
			net_weight = node_mean_fn.keywords['c_node_confound']
			net_weight *= node_mean_fn.keywords['parent_weights_confound_dict'][confound_node]
			B_full[confound_node, node] = net_weight

		for parent_node in node_mean_fn.keywords['parent_weights_obs_dict'].keys():
			net_weight = node_mean_fn.keywords['c_node_signal']
			net_weight *= node_mean_fn.keywords['parent_weights_obs_dict'][parent_node]
			B_full[parent_node, node] = net_weight

		intercepts[node] = node_mean_fn.keywords['intercept']

	B_obs = B_full[num_confounders:, num_confounders:]
	B_latent = B_full[:num_confounders, num_confounders:]
	intercepts_obs = intercepts[num_confounders:]
	confounder_idcs = list(range(num_confounders))

	true_latent_component = (np.linalg.inv((np.eye(p) - B_obs)).T.dot(B_latent.T.dot(data_full[:, confounder_idcs].T))).T
	true_latent_component += intercepts_obs

	return np.mean((true_latent_component - pcss_suff_stats) ** 2, axis=0)

# Load in simulated data
with open('../data/synthetic/pcss_eval_data.pkl', 'rb') as handle:
    pcss_eval_data = pickle.load(handle)

# Load in simulated data configurations
p_grid = [250, 500, 1000]
N_grid = [125, 250, 500]
p_N_grid = list(zip(p_grid, N_grid))
num_confounders = 1
settings = {'linear', 'non_linear'}
basis_fns = dict()
basis_fns['linear'] = [linear_trend]
basis_fns['non_linear'] = [linear_trend, sin_trend, trunctated_quadratic_trend]
noise_var = .2
hidden_r2 = [.25, .5, .75]
expected_neighbs = 5
confounder_pervasiveness = .7
N_sims = 25

# PCSS Performance
pcss_performance = dict()
for setting in settings:
	for p_N in p_N_grid:
		p, N = p_N
		for confound_r2 in hidden_r2:
			for sim_ix in range(N_sims):
				config = (sim_ix, setting, p, N, confound_r2)
				pcss_config = pcss_eval_data[config]
				if setting == 'linear':
					pcss_performance[config] = compute_mse_linear(pcss_config[1], pcss_config[0], pcss_config[2][0], num_confounders)
				else:
					pcss_performance[config] = compute_mse_non_linear(pcss_config[1], pcss_config[0], pcss_config[2][0], num_confounders)


# Save the results
with open('../results/synthetic/pcss_eval_results.pkl', 'wb') as handle:
    pickle.dump(pcss_performance, handle, protocol=pickle.HIGHEST_PROTOCOL)

