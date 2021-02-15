import pickle
import os
from time import time
import numpy as np
# from utils.misc import score_candidate_dags
from models.baseline_models import compute_score_parent_set
from models.all_models import GP_mll_parent_scorer
from decamfound.pcss import get_confounding_suff_stats
from multiprocessing import Pool, cpu_count

RESULT_FILE = '../results/synthetic/eval_removal_parent_set_results_lrps.pkl'
RESULT_FILE_TEMP = '../results/synthetic/eval_removal_parent_set_results_lrps_temp.pkl'

num_parent_sets = 100
graph_eval_data_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'synthetic', 'parent_set_removal_eval_data.pkl')

start = time()
graph_eval_data = pickle.load(open(graph_eval_data_filename, 'rb'))
print(f"Took {time() - start} seconds to load")

VANILLA_BIC = False
POET = False
LRPS = True
CAM = False
CAM_CHEAT = False
DECAMFOUND = False

OVERWRITE = False
if os.path.exists(RESULT_FILE) and not OVERWRITE:
    result_dict = pickle.load(open(RESULT_FILE, 'rb'))
else:
    result_dict = dict()

for (sim_num, setting, p, n, strength), sim_data in graph_eval_data.items():
    print(f"========= setting: {setting}, p={p}, n={n}, strength={strength}, sim={sim_num} ========")
    if (sim_num, setting, p, n, strength) in result_dict:
        continue

    num_confounders = sim_data['num_confounders']
    data_obs = sim_data['data_full'][:, num_confounders:]
    cov = np.cov(data_obs, rowvar=False)
    suffstat = dict(cov=cov, nsamples=data_obs.shape[0], samples=data_obs)

    node = sim_data['random_node']
    candidate_parent_sets = sim_data['restricted_parent_sets'][:num_parent_sets]
    effective_rank = sim_data['effective_rank']
    G_true = sim_data['G_true']
    true_parent_set = G_true.parents_of(node)

    result_dict[(sim_num, setting, p, n, strength)] = dict()
    result_dict[(sim_num, setting, p, n, strength)]['candidate_parent_sets'] = candidate_parent_sets

    # run scoring for covariance
    if VANILLA_BIC:
        try:
            vanilla_scores, vanilla_true_score = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'normal')
            percent_better_than_true = np.mean(vanilla_scores > vanilla_true_score)
            print("Vanilla BIC:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['vanilla_scores'] = vanilla_scores
            result_dict[(sim_num, setting, p, n, strength)]['vanilla_true_score'] = vanilla_true_score
        except:
            print('ERROR')
            continue

    if POET:
        try:
            poet_scores, poet_true_score = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'poet', K=effective_rank)
            percent_better_than_true = np.mean(poet_scores > poet_true_score)
            print("POET+BIC:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['poet_scores'] = poet_scores
            result_dict[(sim_num, setting, p, n, strength)]['poet_true_score'] = poet_true_score
        except:
            print('ERROR')
            continue
    if LRPS:
        try:
            lrps_scores, lrps_true_score = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'lrps', lambda1=5, lambda2=5)
            percent_better_than_true = np.mean(lrps_scores > lrps_true_score)
            print("LRpS+BIC:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['lrps_scores'] = lrps_scores
            result_dict[(sim_num, setting, p, n, strength)]['lrps_true_score'] = lrps_true_score
        except:
            print('ERROR')
            continue

    if CAM:
        try:
            pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
            suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
            cam_scores, cam_true_score = GP_mll_parent_scorer('CAM', suffstats, node, candidate_parent_sets, true_parent_set)
            percent_better_than_true = np.mean(cam_scores > cam_true_score)
            print("CAM:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['cam_scores'] = cam_scores
            result_dict[(sim_num, setting, p, n, strength)]['cam_true_score'] = cam_true_score
        except:
            print('ERROR')
            continue

    if CAM_CHEAT:
        try:
            pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
            suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
            cam_cheat_scores, cam_cheat_true_score = GP_mll_parent_scorer('CAM-CHEAT', suffstats, node, candidate_parent_sets, true_parent_set)
            percent_better_than_true = np.mean(cam_cheat_scores > cam_cheat_true_score)
            print("CAM-CHEAT:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['cam_cheat_scores'] = cam_cheat_scores
            result_dict[(sim_num, setting, p, n, strength)]['cam_cheat_true_score'] = cam_cheat_true_score
        except:
            print('ERROR')
            continue

    if DECAMFOUND:
        try:
            pcss = get_confounding_suff_stats(data_obs, K=effective_rank)[0]
            suffstats = dict(obs_data=data_obs, true_confounders=sim_data['data_full'][:, :num_confounders], pcss=pcss)
            decamfound_scores, decamfound_true_score = GP_mll_parent_scorer('decamfounder', suffstats, node, candidate_parent_sets, true_parent_set)
            percent_better_than_true = np.mean(decamfound_scores > decamfound_true_score)
            print("DECAMFOUND:", percent_better_than_true)

            result_dict[(sim_num, setting, p, n, strength)]['decamfound_scores'] = decamfound_scores
            result_dict[(sim_num, setting, p, n, strength)]['decamfound_true_score'] = decamfound_true_score
        except:
            print('ERROR')
            continue

    pickle.dump(result_dict, open(RESULT_FILE_TEMP, 'wb'))


pickle.dump(result_dict, open(RESULT_FILE, 'wb'))
