import numpy as np
from numpy.linalg import lstsq


def bic_score_parent_set(node, parents, cov, nsamples):
    parents = list(parents)
    var = cov[node, node] - cov[node, parents] @ lstsq(cov[np.ix_(parents, parents)], cov[parents, node], rcond=None)[0]
    penalty = (.5 * np.log(nsamples) * (len(parents) + 2))
    if abs(var) < 1e-10:
        return -penalty
    ll_node = -(nsamples / 2 * np.log(2*np.pi*var) + nsamples / 2)
    return ll_node - penalty


def bic_score(dag, cov, nsamples):
    log_likelihood = 0
    node_mlls = []
    for node in dag.topological_sort():
        parents = list(dag.parents_of(node))
        # compute the conditional covariance of the node given its parents, using Schur complement
        var = cov[node, node] - cov[node, parents] @ lstsq(cov[np.ix_(parents, parents)], cov[parents, node], rcond=None)[0]
        # var_ = 1/np.linalg.inv(cov[np.ix_([node, *parents], [node, *parents])])[0, 0]
        # print(var, var_)
        ll_node = -(nsamples / 2 * np.log(2*np.pi*var) + nsamples / 2)
        log_likelihood += ll_node
        node_mlls.append(ll_node - (.5 * np.log(nsamples) * (len(parents) + 2)))
    penalty_term = .5 * np.log(nsamples) * (dag.num_arcs + 2 * dag.nnodes)  # I've tried with just # arcs + # nodes
    return log_likelihood - penalty_term, np.array(node_mlls)


## FUNCTION TO CHECK CALCULATION OF LL
# def gloglik(samples, node, cond_set, cov, nsamples):
#     var = cov[node, node] - cov[node, cond_set] @ lstsq(cov[np.ix_(cond_set, cond_set)], cov[cond_set, node], rcond=None)[0]
#     ll_node = -(nsamples / 2 * np.log(2*np.pi*var) + nsamples / 2)
#
#     if samples is not None:
#         from sklearn.linear_model import LinearRegression
#         lr = LinearRegression()
#         X = samples[:, cond_set]
#         y = samples[:, node]
#         lr.fit(X, y)
#         residuals = y - X @ lr.coef_ - lr.intercept_
#         from scipy.stats import norm
#
#         ll_nodes2 = norm.logpdf(residuals, scale=np.std(residuals))
#         ll2 = sum(ll_nodes2)
#         print(ll2)
#     return ll_node


def update_bic_score(new_dag, score, cov, nsamples, update):
    update_type, i, j = update

    if update_type == 'add' or update_type == 'remove':
        parents = list(new_dag.parents_of(j))
        neg_log_var = cov[j, j] - cov[j, parents] @ cov[np.ix_(parents, parents)] @ cov[parents, j]

        # TODO NEED TO CHANGE PENALTY TERM
    # update is reversal
    else:
        parents_j = list(new_dag.parents_of(j))
        neg_log_var_j = cov[j, j] - cov[j, parents_j] @ cov[np.ix_(parents_j, parents_j)] @ cov[parents_j, j]
        parents_i = list(new_dag.parents_of(i))
        neg_log_var_j = cov[i, i] - cov[i, parents_i] @ cov[np.ix_(parents_i, parents_i)] @ cov[parents_i, i]

    return score


if __name__ == '__main__':
    import causaldag as cd
    from paper_experiments.code.R_algs import bic_score_R, bic_score_bnlearn_R

    d = cd.rand.directed_erdos(10, .5)
    g = cd.rand.rand_weights(d)
    nsamples = 10
    s = g.sample(nsamples)
    cov = np.cov(s, rowvar=False)
    score = bic_score(d, cov, nsamples)
    scoreR = bic_score_R(d, s)
    score_bnlearn_R = bic_score_bnlearn_R(d, s)
    print("our score, base e", score)
    print("pcalg score", scoreR)
    print("bnlearn score", score_bnlearn_R)

    # TODO: CHECK AGAINST R
    s = g.sample(100)
    prec_matrix = np.linalg.inv(np.cov(s, rowvar=False))
    # score = bic_score(d, prec_matrix)
