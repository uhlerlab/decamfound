from decamfound.scores import log_marginal_like_MAP
import torch
import numpy as np
from tqdm import tqdm
from torch.distributions.normal import Normal


def GP_mll_parent_scorer(score_type, suffstats, node, 
                        candidate_parent_sets, true_parent_set=None, 
                        **kwargs):

    data_obs = torch.tensor(suffstats['obs_data'])
    N = data_obs.shape[0]
    X_node = data_obs[:, node]
    true_confounders = torch.tensor(suffstats['true_confounders'])
    pcss_suff_stats = torch.tensor(suffstats['pcss'])

    if score_type == 'decamfounder':
        parent_scorer = lambda parent_set: log_marginal_like_MAP(data_obs[:, parent_set], 
                                                                 pcss_suff_stats[:, parent_set + [node]], 
                                                                 X_node - pcss_suff_stats[:, node], 
                                                                 include_node=False, **kwargs)[0]

    elif score_type == 'CAM':
        parent_scorer = lambda parent_set: log_marginal_like_MAP(data_obs[:, parent_set], 
                                                                 torch.zeros((N, 0)), 
                                                                 X_node, include_node=False, **kwargs)[0]


    elif score_type == 'CAM-CHEAT':
        parent_scorer = lambda parent_set: log_marginal_like_MAP(data_obs[:, parent_set], 
                                                                 true_confounders, 
                                                                 X_node, include_node=False, **kwargs)[0]

    else:
        raise NotImplementedError

    parent_set_mlls = []
    if len(candidate_parent_sets) < 25:
        parent_set_iterator = candidate_parent_sets
    else:
        parent_set_iterator = tqdm(candidate_parent_sets)
    for parent_set in parent_set_iterator:
        if len(parent_set) == 0:
            X_node = data_obs[:, node]
            if score_type == 'CAM':
                est_noise_sd = X_node.std().item() # MLE estimate of noise
                score = Normal(loc=0, scale=est_noise_sd).log_prob(X_node).mean().item()
            elif score_type == 'decamfounder':
                X_node_resid = X_node - pcss_suff_stats[:, node] # For a source node, s_j - r_j = s_j = pcss[node]
                est_noise_sd = X_node_resid.std().item()
                score = Normal(loc=0, scale=est_noise_sd).log_prob(X_node_resid).mean().item()
            elif score_type == 'CAM-CHEAT':
                score = log_marginal_like_MAP(true_confounders, torch.zeros((N, 0)), X_node, include_node=False, **kwargs)[0]
            parent_set_mlls.append(score)
        else:
            parent_set_mlls.append(parent_scorer(parent_set))

    if true_parent_set is not None:
        if len(true_parent_set) != 0:
            true_parent_set_mll = parent_scorer(list(true_parent_set))
        else:
            if score_type == 'CAM':
                est_noise_sd = X_node.std().item() # MLE estimate of noise
                true_parent_set_mll = Normal(loc=0, scale=est_noise_sd).log_prob(X_node).mean().item()
            elif score_type == 'decamfounder':
                X_node_resid = X_node - pcss_suff_stats[:, node] # For a source node, s_j - r_j = s_j = pcss[node]
                est_noise_sd = X_node_resid.std().item()
                true_parent_set_mll = Normal(loc=0, scale=est_noise_sd).log_prob(X_node_resid).mean().item()
            elif score_type == 'CAM-CHEAT':
                true_parent_set_mll = log_marginal_like_MAP(true_confounders, torch.zeros((N, 0)), X_node, include_node=False, **kwargs)[0]

        return N*np.array(parent_set_mlls), N*true_parent_set_mll
    return N*np.array(parent_set_mlls)


def GP_mll_parent_scorer_dag_helper(suffstats, node_parent_set_tuples, method='CAM',
                        **kwargs):

    data_obs = torch.tensor(suffstats['obs_data'])
    N = data_obs.shape[0]
    true_confounders = torch.tensor(suffstats['true_confounders'])
    pcss_suff_stats = torch.tensor(suffstats['pcss'])

    if method == 'decamfounder':
        parent_scorer = lambda node, parent_set: log_marginal_like_MAP(data_obs[:, parent_set],
                                                                 pcss_suff_stats[:, parent_set + [node]],
                                                                 data_obs[:, node] - pcss_suff_stats[:, node],
                                                                 include_node=False, **kwargs)[0]

    elif method == 'CAM':
        parent_scorer = lambda node, parent_set: log_marginal_like_MAP(data_obs[:, parent_set],
                                                                 torch.zeros((N, 0)),
                                                                 data_obs[:, node], include_node=False, **kwargs)[0]


    elif method == 'CAM-CHEAT':
        parent_scorer = lambda node, parent_set: log_marginal_like_MAP(data_obs[:, parent_set],
                                                                 true_confounders,
                                                                 data_obs[:, node], include_node=False, **kwargs)[0]

    else:
        raise NotImplementedError

    parent_set_mlls = dict()
    if len(node_parent_set_tuples) < 25:
        parent_set_iterator = node_parent_set_tuples
    else:
        parent_set_iterator = tqdm(node_parent_set_tuples)
    for node, parent_set in parent_set_iterator:
        if len(parent_set) == 0:
            X_node = data_obs[:, node]
            if method == 'CAM':
                est_noise_sd = X_node.std().item() # MLE estimate of noise
                score = Normal(loc=0, scale=est_noise_sd).log_prob(X_node).mean().item()
            elif method == 'decamfounder':
                X_node_resid = X_node - pcss_suff_stats[:, node] # For a source node, s_j - r_j = s_j = pcss[node]
                est_noise_sd = X_node_resid.std().item()
                score = Normal(loc=0, scale=est_noise_sd).log_prob(X_node_resid).mean().item()
            elif method == 'CAM-CHEAT':
                score = log_marginal_like_MAP(true_confounders, torch.zeros((N, 0)), X_node, include_node=False, **kwargs)[0]
        else:
            score = parent_scorer(node, sorted(list(parent_set)))
        parent_set_mlls[(node, parent_set)] = N*score

    return parent_set_mlls
