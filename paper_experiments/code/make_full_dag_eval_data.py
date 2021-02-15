
import numpy as np
import causaldag as cd

from decamfound.pcss import get_confounding_suff_stats
from utils.generate_synthetic_data import make_confound_data
from causaldag import DAG, CamDAG
from utils.basis_fns import *
from utils.sample_dag_neighborhood import add_remove_dag_sampler, sample_dag_neighborhood

import pickle
import random
import numpy as np

np.random.seed(12098312)
random.seed(12098312)

p_grid = [500]
N_grid = [250]
p_N_grid = list(zip(p_grid, N_grid))
num_confounders = 1
settings = {
	'linear',
	'non_linear'
}
basis_fns = dict()
basis_fns['linear'] = [linear_trend]
basis_fns['non_linear'] = [sin_trend, trunctated_quadratic_trend]
noise_var = .2
hidden_r2 = [.5]
expected_neighbs = 5
confounder_pervasiveness = .7

N_sims = 10
N_rand_dag_samps = 100

graph_eval_data = dict()

for setting in settings:
	for p_N in p_N_grid:
		p, N = p_N
		for confound_r2 in hidden_r2:
			for sim_ix in range(N_sims):
				print((setting, p, confound_r2, sim_ix))
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)] = dict()

				# Generate random DAG
				dag = cd.rand.directed_erdos_with_confounders(p, exp_nbrs=expected_neighbs,
											num_confounders=num_confounders, random_order=False,
											confounder_pervasiveness=confounder_pervasiveness)

				confound_basis = basis_fns[setting]
				signal_basis = basis_fns[setting]
				signal_var = 1 - confound_r2 - noise_var
				confound_cam = make_confound_data(dag, num_confounders, confound_basis,
										signal_basis, signal_var, confound_r2, num_monte_carlo=1000)
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['effective_rank'] = len(confound_basis) * num_confounders
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['num_confounders'] = num_confounders

				# Sample full dataset
				data_full = confound_cam.sample(N)
				data_obs = data_full[:, num_confounders:]

				# Save the DAG and sampled datapoints
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['data_full'] = data_full
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['confound_cam'] = confound_cam

				observable_est = set(range(num_confounders, num_confounders+p))
				d_subgraph = dag.induced_subgraph(observable_est)
				G_true = d_subgraph.rename_nodes({i: i-num_confounders for i in observable_est})
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['G_true'] = G_true

				# Randomly sample graphs
				rand_graphs = add_remove_dag_sampler(G_true, nsamples=N_rand_dag_samps)
				# rand_graphs = sample_dag_neighborhood(G_true, add_prob=0, reverse_prob=0, nsamples=N_rand_dag_samps)
				graph_eval_data[(sim_ix, setting, p, N, confound_r2)]['rand_graphs'] = rand_graphs

# Save the DAGs and sampled datapoints
with open('../data/synthetic/mixed_graph_eval_data.pkl', 'wb') as handle:
    pickle.dump(graph_eval_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
