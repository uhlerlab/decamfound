
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(style='white',)

import sys
sys.path.append("../code/")

from utils.metrics import dag_sample_metrics

with open('../results/synthetic/eval_mixed_dag_results.pkl', 'rb') as handle:
    eval_dag_results = pickle.load(handle)

dag_shd_max = dict()
dag_shd_avg = dict()

METHODS = ['PCSS+BIC', 'Vanilla BIC', 'CAM', 'LRPS+BIC', 'CAM-OBS', 'DeCAMFound']
DICT_ROOTS = ['poet', 'vanilla', 'cam', 'lrps', 'cam_cheat', 'decamfound']
methods2dict = dict(zip(METHODS, DICT_ROOTS))

for (sim_num, setting, p, n, strength), sim_data in eval_dag_results.items():
		for method in METHODS:
			dags = sim_data['candidate_dags']
			method_root = methods2dict[method]
			dag_scores = sim_data['{0}_scores'.format(method_root)][0]
			shd_best, avg_weighted_shd = dag_sample_metrics(dags, 
														dag_scores)

			if (method, setting, p, n, strength) not in dag_shd_max.keys():
				dag_shd_max[(method, setting, p, n, strength)] = []
				dag_shd_avg[(method, setting, p, n, strength)] = []
			
			dag_shd_max[(method, setting, p, n, strength)].append(round(shd_best, 5))
			dag_shd_avg[(method, setting, p, n, strength)].append(round(avg_weighted_shd, 5))



performance_df = []
N_sims = len(list(dag_shd_max.values())[0])

for (method, setting, p, n, strength) in dag_shd_max.keys():
	shd_maxs = dag_shd_max[(method, setting, p, n, strength)]
	shd_avgs = dag_shd_avg[(method, setting, p, n, strength)]
	for i in range(N_sims):
		if setting == 'linear':
			performance_df.append([method, 'Linear', p, n, strength, 
								shd_maxs[i], shd_avgs[i]])
		else:
			performance_df.append([method, 'Non-Linear', p, n, strength, 
								shd_maxs[i], shd_avgs[i]])



performance_df = pd.DataFrame(performance_df, columns=['Method', 'Setting', 'p', 'N', 
													   'Confound R^2', 'SHD of MAP', 
													   'Post. Avg. SHD'])


linear_mask = (performance_df.Setting == 'linear').values


fig, ax = plt.subplots(1, figsize=(7, 4), dpi=200)
sns_box = sns.boxplot(x='Method',y='SHD of MAP', data=performance_df, 
					 ax=ax, hue='Setting')
# ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('SHD Between MAP and True DAG', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/dag_shd.png')
plt.close()


fig, ax = plt.subplots(1, figsize=(7, 4), dpi=200)
sns_box = sns.boxplot(x='Method',y='Post. Avg. SHD', data=performance_df, 
					 ax=ax, hue='Setting')
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Avg. Posterior SHD', fontsize=12)
plt.tight_layout()
plt.savefig('../figures/dag_mll_mag.png')
plt.close()


