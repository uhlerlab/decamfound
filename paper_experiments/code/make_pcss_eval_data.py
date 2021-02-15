
import numpy as np
import causaldag as cd

from decamfound.pcss import get_confounding_suff_stats
from utils.generate_synthetic_data import make_confound_data
from causaldag import DAG, CamDAG
from utils.basis_fns import *

import pickle

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

pcss_eval_data = dict()

for setting in settings:
	for p_N in p_N_grid:
		p, N = p_N
		for confound_r2 in hidden_r2:
			for sim_ix in range(N_sims):
				print((setting, p, confound_r2, sim_ix))
				# Generate random DAG
				dag = cd.rand.directed_erdos_with_confounders(p, exp_nbrs=expected_neighbs, 
											num_confounders=num_confounders, random_order=False,
											confounder_pervasiveness=confounder_pervasiveness)

				confound_basis = basis_fns[setting]
				signal_basis = basis_fns[setting]
				signal_var = 1 - confound_r2 - noise_var
				confound_cam = make_confound_data(dag, num_confounders, confound_basis, 
										signal_basis, signal_var, confound_r2, num_monte_carlo=1000)

				# Sample full dataset
				data_full = confound_cam.sample(N)
				data_obs = data_full[:, num_confounders:]

				# Compute pcss suff stats
				pcss = get_confounding_suff_stats(data_obs, K=len(basis_fns[setting]) * num_confounders)

				# Save the DAG and sampled datapoints
				pcss_eval_data[(sim_ix, setting, p, N, confound_r2)] = [data_full, confound_cam, pcss]


# Save the DAGs and sampled datapoints
with open('../data/synthetic/pcss_eval_data.pkl', 'wb') as handle:
    pickle.dump(pcss_eval_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

