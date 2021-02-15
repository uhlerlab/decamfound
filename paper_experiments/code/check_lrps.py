import pickle
import os
from time import time
import numpy as np
# from utils.misc import score_candidate_dags
from models.baseline_models import compute_score_parent_set_dag_helper
from models.all_models import GP_mll_parent_scorer_dag_helper
from utils.misc import score_candidate_dags
from utils.metrics import dag_sample_metrics
from decamfound.pcss import get_confounding_suff_stats
from multiprocessing import Pool, cpu_count
from functools import partial
import random
from utils.sample_dag_neighborhood import sample_dag_neighborhood

import causaldag as cd

d = cd.rand.directed_erdos_with_confounders(10, .2)
g = cd.rand.rand_weights(d)
nsamples = 1000
samples = g.sample(nsamples)
data_obs = samples[:, 1:]
suffstat = dict(samples=data_obs, cov=np.cov(samples, rowvar=False), nsamples=nsamples)

obs_nodes = set(range(1, 11))
d_true = d.induced_subgraph(obs_nodes)
d_true = d_true.rename_nodes({i: i-1 for i in obs_nodes})
candidate_dags, _ = sample_dag_neighborhood(d_true)
candidate_dag_scores_lrps, _ = score_candidate_dags(suffstat, candidate_dags, compute_score_parent_set_dag_helper, method='lrps')
candidate_dag_scores_bic, _ = score_candidate_dags(suffstat, candidate_dags, compute_score_parent_set_dag_helper, method='normal')

print("--- lrps ---")
print(candidate_dag_scores_lrps[0])
print(np.mean(candidate_dag_scores_lrps))
print(np.max(candidate_dag_scores_lrps))

print("--- bic ---")
print(candidate_dag_scores_bic[0])
print(np.mean(candidate_dag_scores_bic))
print(np.max(candidate_dag_scores_bic))
