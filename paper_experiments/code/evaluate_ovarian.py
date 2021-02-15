
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


ALL_OVARIAN_CACHE = pickle.load(open('../results/ALL_OVARIAN_CACHE.pkl', 'rb'))
gene_data = ALL_OVARIAN_CACHE['gene_data']
node_parent_set_tuples = ALL_OVARIAN_CACHE['node_parent_set_tuples']
node_pair_del_tuples = ALL_OVARIAN_CACHE['node_pair_del_tuples']

data_obs = gene_data.values.copy()
data_full = ALL_OVARIAN_CACHE['data_full']
N_tfs = 15
cov_obs = np.cov(data_obs, rowvar=False)
bic_suffstat = dict(cov=cov_obs, nsamples=data_obs.shape[0], samples=data_obs)
cov_lrps_path = ALL_OVARIAN_CACHE['cov_lrps_path']


# Wrong Parent Addition tasks
all_wrong = dict()

vanilla_results_wrong_add = compute_score_parent_set_dag_helper(bic_suffstat, node_parent_set_tuples, method='normal', progress=False)
poet_results_wrong_add = compute_score_parent_set_dag_helper(bic_suffstat, node_parent_set_tuples, method='poet', K=7, progress=False)

lrps_wrong_add_results_dict = dict()
for i in range(len(cov_lrps_path)):
	bic_suffstat_lrps = dict(cov=cov_lrps_path[i], nsamples=data_obs.shape[0], samples=data_obs)
	lrps_wrong_results = compute_score_parent_set_dag_helper(bic_suffstat_lrps, node_parent_set_tuples, method='normal', progress=False)
	lrps_wrong_add_results_dict[i] = lrps_wrong_results


all_wrong['vanilla_wrong_add'] = vanilla_results_wrong_add
all_wrong['poet_wrong_add'] = poet_results_wrong_add
all_wrong['lrps_wrong_add'] = lrps_wrong_add_results_dict

OLD_cached_results = pickle.load(open('../results/real/ovarian_parent_scores.pkl', 'rb'))

all_wrong['cam_wrong_add'] = OLD_cached_results['cam']
all_wrong['cam_cheat_wrong_add'] = OLD_cached_results['cam_cheat']
all_wrong['decamfound_wrong_add'] = OLD_cached_results['decamfound']

pickle.dump(all_wrong, open('../results/real/all_wrong_parent_add.pkl', 'wb'))


# Correct Parent Deletion tasks
all_correct = dict()
vanilla_results_del = compute_score_parent_set_dag_helper(bic_suffstat, node_pair_del_tuples, method='normal', progress=False)
poet_results_del = compute_score_parent_set_dag_helper(bic_suffstat, node_pair_del_tuples, method='poet', K=7, progress=False)

lrps_del_dict = dict()
for i in range(len(cov_lrps_path)):
	bic_suffstat_lrps = dict(cov=cov_lrps_path[i], nsamples=data_obs.shape[0], samples=data_obs)
	lrps_del_results = compute_score_parent_set_dag_helper(bic_suffstat_lrps, node_pair_del_tuples, method='normal', progress=False)
	lrps_del_dict[i] = lrps_del_results


all_correct['vanilla_correct_del'] = vanilla_results_del
all_correct['poet_correct_del'] = poet_results_del
all_correct['lrps_correct_del'] = lrps_del_dict


pcss_suff_stats, est_factor_component, K, evecs, evals = get_confounding_suff_stats(gene_data.values, K=7)
gp_suffstats = dict(obs_data=data_obs, true_confounders=data_full.values[:, :N_tfs], pcss=pcss_suff_stats)

decam_results_del = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_pair_del_tuples, method='decamfounder')
cam_cheat_results_del = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_pair_del_tuples, method='CAM-CHEAT')
cam_results_del = GP_mll_parent_scorer_dag_helper(gp_suffstats, node_pair_del_tuples, method='CAM')

all_correct['decamfound_correct_del'] = decam_results_del
all_correct['cam_cheat_correct_del'] = cam_cheat_results_del
all_correct['cam_correct_del'] = cam_results_del

pickle.dump(all_correct, open('../results/real/all_correct_del.pkl', 'wb'))

