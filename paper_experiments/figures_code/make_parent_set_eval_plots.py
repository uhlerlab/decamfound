
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(style='white',)

import sys
sys.path.append("../code/")

from utils.metrics import parent_set_metrics

with open('../results/synthetic/eval_parent_set_results.pkl', 'rb') as handle:
    eval_parent_set_results = pickle.load(handle)

parent_set_prop_higher = dict()
parent_set_amount_larger = dict()

METHODS = ['PCSS+BIC', 'Vanilla BIC', 'CAM', 'LRPS+BIC', 'CAM-OBS', 'DeCAMFound']
DICT_ROOTS = ['poet', 'vanilla', 'cam', 'lrps', 'cam_cheat', 'decamfound']
methods2dict = dict(zip(METHODS, DICT_ROOTS))

for (sim_num, setting, p, n, strength), sim_data in eval_parent_set_results.items():
		for method in METHODS:
			method_root = methods2dict[method]
			
			parent_set_scores = sim_data['{0}_scores'.format(method_root)]
			true_parent_set_score = sim_data['{0}_true_score'.format(method_root)]
			prop_higher, mag_larger = parent_set_metrics(parent_set_scores, 
														true_parent_set_score)

			if (method, setting, p, n, strength) not in parent_set_prop_higher.keys():
				parent_set_prop_higher[(method, setting, p, n, strength)] = []
				parent_set_amount_larger[(method, setting, p, n, strength)] = []
			
			parent_set_prop_higher[(method, setting, p, n, strength)].append(prop_higher)
			parent_set_amount_larger[(method, setting, p, n, strength)].append(mag_larger)


performance_df = []
N_sims = len(list(parent_set_prop_higher.values())[0])

for (method, setting, p, n, strength) in parent_set_prop_higher.keys():
	prop_higher_all = parent_set_prop_higher[(method, setting, p, n, strength)]
	mag_diff_all = parent_set_amount_larger[(method, setting, p, n, strength)]
	for i in range(N_sims):
		performance_df.append([method, setting, p, n, strength, 
							prop_higher_all[i], mag_diff_all[i]])


performance_df = pd.DataFrame(performance_df, columns=['Method', 'setting', 'p', 'N', 
													   'Confound R^2', 'Prop. Larger', 
													   'MLL Diff.'])


linear_mask = (performance_df.setting == 'linear').values


fig, ax = plt.subplots(1, figsize=(6, 4), dpi=200)
sns_box = sns.boxplot(x='Confound R^2',y='Prop. Larger', data=performance_df[linear_mask], 
			hue='Method', ax=ax)
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns_box.legend_.remove()

ax.set_xlabel('Confound $R^2$', fontsize=12)
ax.set_ylabel('Prop. Times MLL Wrong > MLL True', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/parent_set_linear_prop_metric.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(6, 4), dpi=200)
sns_box = sns.boxplot(x='Confound R^2',y='Prop. Larger', data=performance_df[~linear_mask], 
			hue='Method', ax=ax)
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set_xlabel('Confound $R^2$', fontsize=12)
ax.set_ylabel('Prop. Times MLL Wrong > MLL True', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/parent_set_non_linear_prop_metric.png')
plt.close()



fig, ax = plt.subplots(1, figsize=(6, 4), dpi=200)
sns_box = sns.boxplot(x='Confound R^2',y='MLL Diff.', data=performance_df[linear_mask], 
			hue='Method', ax=ax)
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns_box.legend_.remove()

ax.set_xlabel('Confound $R^2$', fontsize=12)
ax.set_ylabel('Log Odds (Wrong vs. True)', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/parent_set_linear_mll_diff.png')
plt.close()

fig, ax = plt.subplots(1, figsize=(6, 4), dpi=200)
sns_box = sns.boxplot(x='Confound R^2',y='MLL Diff.', data=performance_df[~linear_mask], 
			hue='Method', ax=ax)
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns_box.legend_.remove()
ax.set_xlabel('Confound $R^2$', fontsize=12)
ax.set_ylabel('Log Odds (Wrong vs. True)', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/parent_set_non_linear_mll_diff.png')
plt.close()

