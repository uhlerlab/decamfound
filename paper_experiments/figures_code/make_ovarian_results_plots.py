
import numpy as np
import pandas as pd
import pickle 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white',)

import sys
sys.path.append("../code/")

from utils.metrics import compute_true_prob


def get_all_probs(node_parent_set_tuples, result_dict):
    probs = []
    for i in range(0, len(node_parent_set_tuples), 2):
        true_score = result_dict[node_parent_set_tuples[i]]
        wrong_score_key = [result_dict[node_parent_set_tuples[i+1]]]
        probs.append(compute_true_prob(wrong_score_key, true_score))
    return np.array(probs)


ovarian_wrong_parent_scores = pickle.load(open('../results/real/all_wrong_parent_add.pkl', 'rb'))
ovarian_correct_scores = pickle.load(open('../results/real/all_correct_del.pkl', 'rb'))
ALL_OVARIAN_CACHE = pickle.load(open('../results/ALL_OVARIAN_CACHE.pkl', 'rb'))

node_parent_set_tuples = ALL_OVARIAN_CACHE['node_parent_set_tuples']
node_pair_del_tuples = ALL_OVARIAN_CACHE['node_pair_del_tuples']


method_wrong_probs = dict()
method_correct_probs = dict()

METHODS = ['Vanilla BIC', 'PCSS+BIC', 'LRPS+BIC', 'CAM', 'CAM-OBS', 'DeCAMFound']
DICT_ROOTS = ['vanilla', 'poet', 'lrps', 'cam', 'cam_cheat', 'decamfound']
methods2dict = dict(zip(METHODS, DICT_ROOTS))

for method in METHODS:
    result_key = methods2dict[method] + '_wrong_add'
    if result_key != 'lrps_wrong_add':
        method_wrong_probs[method] = get_all_probs(node_parent_set_tuples, ovarian_wrong_parent_scores[result_key])
    else:
        for cv_ix, probs in ovarian_wrong_parent_scores[result_key].items():
            method_wrong_probs['{0}, i={1}'.format(method, cv_ix)] = get_all_probs(node_parent_set_tuples, ovarian_wrong_parent_scores[result_key][cv_ix])

for method in METHODS:
    result_key = methods2dict[method] + '_correct_del'
    if result_key != 'lrps_correct_del':
        method_correct_probs[method] = get_all_probs(node_pair_del_tuples, ovarian_correct_scores[result_key])
    else:
        for cv_ix, probs in ovarian_correct_scores[result_key].items():
            method_correct_probs['{0}, i={1}'.format(method, cv_ix)] = get_all_probs(node_pair_del_tuples, ovarian_correct_scores[result_key][cv_ix])


results_method_wrong = dict()
results_method_correct = dict()
ratios = dict()

lrps_wrong = []
lrps_correct = []

for key in method_wrong_probs.keys():
    prop_wrong = np.mean(method_wrong_probs[key] < .5)
    prop_correct = np.mean(method_correct_probs[key] >= .5)
    results_method_wrong[key] = prop_wrong
    results_method_correct[key] = prop_correct
    ratios[key] = round(prop_correct / prop_wrong, 2)
    if 'LRPS' in key:
        lrps_wrong.append(prop_wrong)
        lrps_correct.append(prop_correct)
        ratios[key] = round(prop_correct / prop_wrong, 2)

print(ratios)

lrps_wrong = np.array(lrps_wrong)
lrps_correct = np.array(lrps_correct)

for key in results_method_wrong.keys():
    frac_wrong = results_method_wrong[key]
    frac_correct = results_method_correct[key]
    if 'LRPS' not in key:
        sns.scatterplot([100*frac_wrong], [100*frac_correct], label=key, s=100, marker='X')


sns.lineplot(100*lrps_wrong[lrps_wrong.argsort()], 100*lrps_correct[lrps_wrong.argsort()], color='black', label='LRPS CV Path', alpha=.5, linewidth=4)
sns.scatterplot([100*results_method_wrong['LRPS+BIC, i=7']], [100*results_method_correct['LRPS+BIC, i=7']], 
                s=100, color='black', label='LRPS+BIC', marker='X')

plt.xlabel('% of Times Wrong Node Added')
plt.ylabel('% of Times Neighbor Kept')

plt.savefig('../figures/ovarian/ovarian_tpr_fpr.png')

