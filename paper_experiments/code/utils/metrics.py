
import itertools as itr
import numpy as np
from causaldag import DAG, CamDAG
from scipy.special import logsumexp

def dag_sample_metrics(dags, scores):
    true_dag = dags[0]

    best_ix = np.argmax(scores)
    shd_best = true_dag.shd(dags[best_ix])
    shds = [true_dag.shd(dag) for dag in dags]
    avg_unweighted_shd = np.mean(shds)
    log_posteriors = scores - logsumexp(scores)
    posteriors = np.exp(log_posteriors)
    avg_weighted_shd = np.sum(posteriors * np.array(shds))
    return shd_best, avg_weighted_shd


def minimal_imap_perm_score(perm, true_dag):
    assert len(set(perm) - set(true_dag.nodes)) == 0, "Need same vertex set"
    d = DAG(nodes=set(perm))
    for i, node1 in enumerate(perm):
        for j, node2 in enumerate(perm):
            if i < j:
                d.add_arc(node1, node2)
    ixs = list(itr.chain.from_iterable(((f, s) for f in range(s)) for s in range(len(perm))))
    for i, j in ixs:
        pi_i, pi_j = perm[i], perm[j]
        S_ij = set(range(np.maximum(i, j))) - {i} - {j}
        is_ci = true_dag.dsep({pi_i}, {pi_j}, S_ij)
        if is_ci:
            d.remove_arc(pi_i, pi_j)
    return d, true_dag.shd(d)


def parent_set_metrics(parent_set_mlls, true_parent_set_mll):
    parent_set_mlls = np.array(parent_set_mlls)
    prop_larger = np.mean(parent_set_mlls > true_parent_set_mll)
    mag_mll_larger = np.max(parent_set_mlls) - true_parent_set_mll
    return prop_larger, mag_mll_larger


def compute_true_prob(scores, true_score):
    all_scores = [true_score] + list(scores)
    log_posteriors = all_scores - logsumexp(all_scores)
    posteriors = np.exp(log_posteriors)
    return posteriors[0]
