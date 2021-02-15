
import numpy as np
from collections import defaultdict
import random

import itertools as itr
from typing import List, Any, Dict
from scipy.special import comb
from tqdm import tqdm
from functools import partial

from causaldag import DAG, CamDAG
from causaldag.rand.graphs import _cam_mean_function, unif_away_zero

RandWeightFn = Any


class NormalNoise(object):
    def __init__(self, noise_var):
        self.noise_var = noise_var

    def __call__(self, size):
        return np.random.normal(0, np.sqrt(self.noise_var), size=size)


def source_node_mean_fn(parent_vals: np.ndarray, parents: list):
    return np.zeros(parent_vals.shape[0])


def _cam_mean_function_combo(
        parent_vals: np.ndarray,
        parents: list,
        confounder_nodes: set,
        c_node_signal: float,
        c_node_confound: float,
        parent_weights_obs_dict: dict(),
        parent_weights_confound_dict: dict(),
        parent2base_obs: dict,
        parent2base_confound: dict,
        intercept: float
):  
    if len(parents) == 0:
        return np.zeros(parent_vals.shape[0])

    confound_mask = np.array([True if _node in confounder_nodes else False for _node in parents])
    num_confounders = np.sum(confound_mask)
    parents_obs = np.array(parents)[~confound_mask]
    parents_confound = np.array(parents)[confound_mask]
    parent_vals_obs = parent_vals[:, ~confound_mask]
    parent_vals_confound = parent_vals[:, confound_mask]

    assert parent_vals.shape[1] == len(parents)
    assert len(parent_weights_obs_dict.keys()) == len(parents_obs)
    assert len(parent_weights_confound_dict.keys()) == len(parents_confound)
    assert all(parent in parent2base_obs for parent in parents_obs)
    assert all(parent in parent2base_confound for parent in parents_confound)
    
    if parents_obs.shape[0] == 0:
        parent_contribs_obs = np.zeros(parent_vals.shape[0])
    else:
        parent_weights_obs = np.array([parent_weights_obs_dict[node] for node in parents_obs])
        parent_contribs_obs = np.array([parent2base_obs[parent](parent_vals_obs[:, ix]) for ix, parent in enumerate(parents_obs)]).T
        parent_contribs_obs = parent_contribs_obs * parent_weights_obs
        parent_contribs_obs = parent_contribs_obs.sum(axis=1)

    if parents_confound.shape[0] == 0:
        parent_contribs_confound = np.zeros(parent_vals.shape[0])
    else:
        parent_weights_confound = np.array([parent_weights_confound_dict[node] for node in parents_confound])
        parent_contribs_confound = np.array([parent2base_confound[parent](parent_vals_confound[:, ix]) for ix, parent in enumerate(parents_confound)]).T
        parent_contribs_confound = parent_contribs_confound * parent_weights_confound
        parent_contribs_confound = parent_contribs_confound.sum(axis=1)

    return c_node_signal * parent_contribs_obs + c_node_confound * parent_contribs_confound + intercept


def make_confound_data(
        dag: DAG,
        num_confounders: int,
        confound_basis: list,
        signal_basis: list,
        signal_var: float,
        confound_var: float,
        rand_weight_fn: RandWeightFn = unif_away_zero,
        num_monte_carlo: int = 10000,
        progress=False
):
    assert (signal_var < 1) & (signal_var > 0)
    assert (confound_var < 1) & (confound_var > 0)
    assert (signal_var + confound_var) < 1
    obs_noise_var = 1 - signal_var - confound_var
    obs_noise_fn =  NormalNoise(obs_noise_var)
    confound_noise_fn =  NormalNoise(1)

    cam_dag = CamDAG(dag._nodes, arcs=dag._arcs)
    top_order = dag.topological_sort()
    sample_dict = dict()

    confound_nodes = set(np.arange(num_confounders))

    # for each node, create the conditional
    node_iterator = top_order if not progress else tqdm(top_order)
    for ix, node in enumerate(node_iterator):
        if node < num_confounders:
            cam_dag.set_mean_function(node, source_node_mean_fn)
            cam_dag.set_noise(node, confound_noise_fn)
            sample_dict[node] = confound_noise_fn(size=num_monte_carlo)

        else:
            obs_parents = list(dag.parents_of(node) - confound_nodes)
            confound_parents = list(dag.parents_of(node) & confound_nodes)
            all_parents = confound_parents + obs_parents

            n_obs_parents = len(obs_parents)
            n_confound_parents = len(confound_parents)

            parent2base_obs = dict(zip(obs_parents, random.choices(signal_basis, k=n_obs_parents)))
            parent2base_confound = dict(zip(confound_parents, random.choices(confound_basis, k=n_confound_parents)))


            parent_weights_obs = rand_weight_fn(size=n_obs_parents)
            parent_weights_confound = rand_weight_fn(size=n_confound_parents)
            parent_weights_obs_dict = dict(zip(obs_parents, parent_weights_obs))
            parent_weights_confound_dict = dict(zip(confound_parents, parent_weights_confound))

            parent_vals_obs = np.array([sample_dict[parent] for parent in obs_parents]).T if n_obs_parents > 0 else np.zeros([num_monte_carlo, 0])
            parent_vals_confound = np.array([sample_dict[parent] for parent in confound_parents]).T if n_confound_parents > 0 else np.zeros([num_monte_carlo, 0])

            c_signal = 1
            c_confound = 1
            intercept = 0
            if n_obs_parents > 0:
                mean_function_no_c_obs = partial(_cam_mean_function, c_node=1, parent_weights=parent_weights_obs, parent2base=parent2base_obs)
                values_from_parents_obs = mean_function_no_c_obs(parent_vals_obs, obs_parents)
                variance_from_parents_obs = np.var(values_from_parents_obs)
                avg_from_parents_obs = np.mean(values_from_parents_obs)
                if n_confound_parents == 0:
                    c_signal = np.sqrt((signal_var + confound_var) / variance_from_parents_obs)
                else:
                    c_signal = np.sqrt(signal_var / variance_from_parents_obs)
                intercept -= c_signal * avg_from_parents_obs
            if n_confound_parents > 0:
                mean_function_no_c_confound = partial(_cam_mean_function, c_node=1, parent_weights=parent_weights_confound, parent2base=parent2base_confound)
                values_from_parents_confound = mean_function_no_c_confound(parent_vals_confound, confound_parents)
                variance_from_parents_confound = np.var(values_from_parents_confound)
                avg_from_parents_confound = np.mean(values_from_parents_confound)
                c_confound = np.sqrt(confound_var / variance_from_parents_confound)
                if n_obs_parents == 0:
                    obs_noise_fn = NormalNoise(1 - confound_var)

                intercept -= c_confound * avg_from_parents_confound

                if np.isnan(c_confound):
                    raise ValueError

            if (n_obs_parents == 0) & (n_confound_parents == 0):
                obs_noise_fn =  NormalNoise(1) # obs. source node

            if (n_obs_parents > 0) & (n_confound_parents > 0):
                cov_signal_confound = np.cov(values_from_parents_confound, values_from_parents_obs)[0, 1]
                c_rescale = np.sqrt((signal_var + confound_var) / (signal_var + confound_var + 2*c_signal*c_confound*cov_signal_confound))
                c_signal *= c_rescale
                c_confound *= c_rescale
                intercept *= c_rescale

            mean_function = partial(_cam_mean_function_combo, confounder_nodes=confound_nodes.copy(), 
                                    c_node_signal=c_signal, c_node_confound=c_confound,
                                    parent_weights_obs_dict=parent_weights_obs_dict, parent_weights_confound_dict=parent_weights_confound_dict,
                                    parent2base_obs=parent2base_obs, parent2base_confound=parent2base_confound,
                                    intercept=intercept)            
            mean_vals = mean_function(np.concatenate([parent_vals_confound, parent_vals_obs], axis=1), all_parents)
            sample_dict[node] = mean_vals + obs_noise_fn(size=num_monte_carlo)

            cam_dag.set_mean_function(node, mean_function)
            cam_dag.set_noise(node, obs_noise_fn)

        obs_noise_fn = NormalNoise(obs_noise_var)

    return cam_dag
