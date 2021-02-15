import numpy as np
from models.bic_score import bic_score_parent_set
from R_algs.lrps import lrps, lrps_cv, lrps_path
from tqdm import tqdm
import ipdb

methods = {'normal', 'poet', 'lrps', 'lrps_path'}


def _remove_top_pcs(cov, K):
    u_orig, s_orig, v_orig = np.linalg.svd(cov)
    u_new = u_orig[:, K:]
    s_new = s_orig[K:]
    v_new = v_orig[K:]
    return (u_new * s_new) @ v_new


def compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, method='normal', progress=False, **kwargs):
    orig_cov = suffstat['cov']
    nsamples = suffstat['nsamples']

    assert method in methods
    if method == 'poet':
        cov = _remove_top_pcs(orig_cov, kwargs['K'])
    elif method == 'lrps':
        # sqrt_diag = np.sqrt(np.diag(cov))
        # corr = cov / sqrt_diag / sqrt_diag[:, None]
        cov = lrps_cv(suffstat['samples'])
        print(cov)
        print(orig_cov)
    elif method == "lrps_path":
        covs = lrps_path(suffstat["samples"])
    elif method == 'normal':
        cov = orig_cov
    else:
        raise ValueError

    true_s = bic_score_parent_set(node, true_parent_set, cov, nsamples)
    scores = []
    r = tqdm(candidate_parent_sets) if progress else candidate_parent_sets
    for parent_set in r:
        s = bic_score_parent_set(node, parent_set, cov, nsamples)
        scores.append(s)

    return np.array(scores), true_s


def compute_score_parent_set_dag_helper(suffstat, node_parent_set_tuples, method='normal', progress=False, **kwargs):
    cov = suffstat['cov']
    nsamples = suffstat['nsamples']

    assert method in methods
    if method == 'poet':
        cov = _remove_top_pcs(cov, kwargs['K'])
    elif method == 'lrps':
        sqrt_diag = np.sqrt(np.diag(cov))
        # corr = cov / sqrt_diag / sqrt_diag[:, None]
        cov = lrps_cv(suffstat['samples'])

    scores = dict()
    r = tqdm(node_parent_set_tuples) if progress else node_parent_set_tuples
    for node, parent_set in r:
        s = bic_score_parent_set(node, parent_set, cov, nsamples)
        scores[(node, parent_set)] = s

    return scores


# if __name__ == '__main__':
#     import causaldag as cd
#     from paper_experiments.code.R_algs import bic_score_R, bic_score_bnlearn_R
#
#     nnodes = 10
#     d = cd.rand.directed_erdos_with_confounders(10, .2, num_confounders=1, confounder_pervasiveness=1)
#     g = cd.rand.rand_weights(d)
#     nsamples = 10
#     all_samples = g.sample(nsamples)
#     obs_samples = all_samples[:, 1:]
#     cov = np.cov(obs_samples, rowvar=False)
#     suffstat = dict(cov=cov, nsamples=nsamples)
#
#     node = 5
#     true_parent_set = d.parents_of(node)
#     true_parent_set = {p-1 for p in true_parent_set}
#     print(f"True parent set: {true_parent_set}")
#     candidate_parent_sets = [true_parent_set | {p} for p in set(range(nnodes)) - {node-1, *true_parent_set}]
#
#     standard_scores, true_score1 = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'normal')
#     poet_scores, true_score2 = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'poet', K=3)
#     lrps_scores, true_score3 = compute_score_parent_set(suffstat, node, candidate_parent_sets, true_parent_set, 'lrps', lambda1=5, lambda2=5)
#
#     standard_val = sum(standard_scores > true_score1)/len(standard_scores)
#     poet_val = sum(poet_scores > true_score2)/len(poet_scores)
#
#     print(f"Standard: {standard_val}")
#     print(f"POET: {poet_val}")
