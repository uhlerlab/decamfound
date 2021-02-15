
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set(style='white',)

import sys
sys.path.append("../code/")


with open('../results/synthetic/pcss_eval_results.pkl', 'rb') as handle:
    pcss_performance = pickle.load(handle)

max_errors = dict()

for (sim_num, setting, p, n, strength), sim_data in pcss_performance.items():
	if (setting, p, n, strength) not in max_errors.keys():
		max_errors[(setting, p, n, strength)] = []
	max_errors[(setting, p, n, strength)].append(np.max(sim_data))

max_error_df = []

for (setting, p, n, strength), errors in max_errors.items():
	for error in errors:
		max_error_df.append([setting, p, n, strength, error])

max_error_df = pd.DataFrame(max_error_df, columns=['setting', 'p', 'N', 'Confound R^2', 'Max MSE'])

############################################ Boxplot ############################################

fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=200)
sns.boxplot(x='p',y='Max MSE', data=max_error_df[max_error_df.setting == 'linear'], 
			hue='Confound R^2', ax=ax[0])
ax[0].set_xlabel('# of Covariates (p)', fontsize=12)
# plt.ylabel('$\max_{i \in [p]} \ MSE(s_i, \hat{s}_i)$', fontsize=12)
ax[0].set_ylabel('Max MSE', fontsize=12)

ax[0].set_title('Linear SEM')

sns.boxplot(x='p',y='Max MSE', data=max_error_df[max_error_df.setting == 'non_linear'], 
			hue='Confound R^2', ax=ax[1])

ax[1].set_xlabel('# of Covariates (p)', fontsize=12)
ax[1].set_ylabel('Max MSE', fontsize=12)
ax[1].set_title('Non-Linear CAM SEM')

plt.tight_layout()

plt.savefig('../figures/pcss_boxplots.png')
plt.close()

############################################ Scatterplot ############################################


# Plot Non-linear Visuals
with open('../data/synthetic/pcss_eval_data.pkl', 'rb') as handle:
    pcss_eval_data = pickle.load(handle)


sim0_result = pcss_eval_data[(0, 'non_linear', 500, 250, 0.5)]
data_full_sim0 = sim0_result[0]
confound_cam_sim0 = sim0_result[1]
pcss_suff_stats_sim0 = sim0_result[2][0]

fig, ax = plt.subplots(2, 2, figsize=(6, 6), dpi=200)
sns.set(style='white',)
sns.scatterplot(data_full_sim0[:, 0], data_full_sim0[:, 2], label='data', marker='x', color='blue', ax=ax[0, 0], alpha=.5)
sns.scatterplot(data_full_sim0[:, 0], pcss_suff_stats_sim0[:, 1], label='pcss', color='green', ax=ax[0, 0])
ax[0, 0].set_xlabel('Latent Confounder $h_1$', fontsize=12)
ax[0, 0].set_ylabel('Observed Variable $x_{i}$', fontsize=12)

sns.scatterplot(data_full_sim0[:, 0], data_full_sim0[:, 125], label='data', marker='x', color='blue', ax=ax[0, 1], alpha=.5)
sns.scatterplot(data_full_sim0[:, 0], pcss_suff_stats_sim0[:, 124], label='pcss', color='green', ax=ax[0, 1])
ax[0, 1].set_xlabel('Latent Confounder $h_1$', fontsize=12)
ax[0, 1].set_ylabel('Observed Variable $x_{i}$', fontsize=12)


sns.scatterplot(data_full_sim0[:, 0], data_full_sim0[:, 250], label='data', marker='x', color='blue', ax=ax[1, 0], alpha=.5)
sns.scatterplot(data_full_sim0[:, 0], pcss_suff_stats_sim0[:, 249], label='pcss', color='green', ax=ax[1, 0])
ax[1, 0].set_xlabel('Latent Confounder $h_1$', fontsize=12)
ax[1, 0].set_ylabel('Observed Variable $x_{i}$', fontsize=12)


sns.scatterplot(data_full_sim0[:, 0], data_full_sim0[:, 500], label='data', marker='x', color='blue', ax=ax[1, 1], alpha=.5)
sns.scatterplot(data_full_sim0[:, 0], pcss_suff_stats_sim0[:, 499], label='pcss', color='green', ax=ax[1, 1])
ax[1, 1].set_xlabel('Latent Confounder $h_1$', fontsize=12)
ax[1, 1].set_ylabel('Observed Variable $x_{i}$', fontsize=12)

plt.tight_layout()
sns.despine()
plt.savefig('../figures/pcss_scatterplot.png')
plt.close()
