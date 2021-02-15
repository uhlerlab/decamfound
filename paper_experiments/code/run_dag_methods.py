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

np.random.seed(12098312)
random.seed(12098312)

RESULT_FILE = '../results/synthetic/eval_mixed_dag_results.pkl'
RESULT_FILE_TEMP = '../results/synthetic/eval_mixed_dag_results_temp.pkl'

graph_eval_data_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'mixed_graph_eval_data.pkl')

start = time()
graph_eval_data = pickle.load(open(graph_eval_data_filename, 'rb'))
print(f"Took {time() - start} seconds to load")

VANILLA_BIC = True
POET = True
LRPS = True
CAM = True
CAM_CHEAT = True
DECAMFOUND = True

OVERWRITE_LRPS = True

OVERWRITE = False
if os.path.exists(RESULT_FILE) and not OVERWRITE:
    result_dict = pickle.load(open(RESULT_FILE, 'rb'))
else:
    result_dict = dict()

for (sim_num, setting, p, n, strength), sim_data in graph_eval_data.items():
    already_exists = (sim_num, setting, p, n, strength) in result_dict
    if already_exists and not OVERWRITE_LRPS:
        continue

    num_confounders = sim_data['num_confounders']
    data_obs = sim_data['data_full'][:, num_confounders:]
    cov = np.cov(data_obs, rowvar=False)
    suffstat = dict(cov=cov, nsamples=data_obs.shape[0], samples=data_obs)

    G_true = sim_data['G_true']
    candidate_dags = [G_true] + sim_data['rand_graphs'][0]
    effective_rank = sim_data['effective_rank']
    setting_results = result_dict[(sim_num, setting, p, n, strength)] if already_exists else dict()
    result_dict[(sim_num, setting, p, n, strength)] = setting_results
    setting_results['candidate_dags'] = candidate_dags

    # run scoring for covariance
    print(f"========= setting: {setting}, p={p}, n={n}, strength={strength}, sim={sim_num} ========")
    if VANILLA_BIC and ('vanilla_scores' not in setting_results):
        candidate_dag_scores = score_candidate_dags(suffstat, candidate_dags, compute_score_parent_set_dag_helper, method='normal')
        setting_results['vanilla_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('VANILLA BIC DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd,post_shd))

    if POET and ('poet_scores' not in setting_results):
        candidate_dag_scores = score_candidate_dags(suffstat, candidate_dags, compute_score_parent_set_dag_helper, method='poet', K=effective_rank)
        setting_results['poet_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('POET BIC DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd,post_shd))

    if LRPS and (OVERWRITE_LRPS or 'lrps_scores' not in setting_results):
        candidate_dag_scores = score_candidate_dags(suffstat, candidate_dags, compute_score_parent_set_dag_helper, method='lrps')
        setting_results['lrps_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('LRPS BIC DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd,post_shd))

    if CAM and ('cam_scores' not in setting_results):
        pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
        suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
        candidate_dag_scores = score_candidate_dags(suffstats, candidate_dags, GP_mll_parent_scorer_dag_helper, method='CAM')
        setting_results['cam_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('CAM DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd,post_shd))

    if CAM_CHEAT and ('cam_cheat_scores' not in setting_results):
        pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
        suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
        candidate_dag_scores = score_candidate_dags(suffstats, candidate_dags, GP_mll_parent_scorer_dag_helper, method='CAM-CHEAT')
        setting_results['cam_cheat_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('CAM-CHEAT DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd,post_shd))

    if DECAMFOUND and ('decamfound_scores' not in setting_results):
        pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
        suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
        candidate_dag_scores = score_candidate_dags(suffstats, candidate_dags, GP_mll_parent_scorer_dag_helper, method='decamfounder')
        setting_results['decamfound_scores'] = candidate_dag_scores
        map_shd, post_shd = dag_sample_metrics(candidate_dags, candidate_dag_scores[0])
        print('DeCAMFound BIC DAG METRICS: MAP-SHD={0}, Avg. Post. SHD={1}'.format(map_shd, post_shd))

    pickle.dump(result_dict, open(RESULT_FILE_TEMP, 'wb')) # Save intermediate results


pickle.dump(result_dict, open(RESULT_FILE, 'wb'))
