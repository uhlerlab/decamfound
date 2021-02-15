
import numpy as np

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV


def get_cheat_component_krr(Xi, latent_confounders):
	regressor = KernelRidge(alpha=1.0)
	param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
	kr = GridSearchCV(KernelRidge(), param_grid=param_grid)
	kr.fit(Xi, latent_confounders)
	return kr.predict(latent_confounders)


def get_cheat_component_random_forest(Xi, latent_confounders):
	regressor = RandomForestRegressor(n_estimators=5000)
	regressor.fit(latent_confounders, Xi)
	return regressor.predict(latent_confounders)


def get_cheat_component_ridgeCV(Xi, latent_confounders):
	regressor = RidgeCV(alphas=np.logspace(-4, 4, 100)).fit(latent_confounders, Xi)
	return regressor.predict(latent_confounders)


def score_candidate_dags(suffstat, candidate_dags, parent_sets_scorer, **kwargs):
	# Get all unique parents
	unique_parent_sets = set()
	for dag in candidate_dags:
		for node in dag.nodes:
			unique_parent_sets.add((node, frozenset(dag.parents_of(node))))

	unique_parent_sets = list(unique_parent_sets)
	print(len(unique_parent_sets))
	parent_set2score = parent_sets_scorer(suffstat, unique_parent_sets, **kwargs)

	dag_scores = []
	for dag in candidate_dags:
		dag_score = 0
		for node in dag.nodes:
			parent_set = frozenset(dag.parents_of(node))
			dag_score += parent_set2score[(node, parent_set)]
		dag_scores.append(dag_score)

	return np.array(dag_scores), parent_set2score


