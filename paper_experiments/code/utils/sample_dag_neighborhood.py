import causaldag as cd
import random
import itertools as itr
import numpy as np
from tqdm import tqdm


def sample_dag_neighborhood(
        dag,
        add_prob=.5,
        reverse_prob=0,
        nsamples=100
):
    assert add_prob + reverse_prob <= 1
    rand_vals = np.random.random(size=nsamples)
    extra_rand_vals = np.random.random(size=nsamples)

    nodes = dag.nodes
    max_arcs = len(nodes) * (len(nodes) - 1)/2
    dags = []
    updates = []
    current_dag: cd.DAG = dag.copy()
    curr_missing_edges = {frozenset({i, j}) for i, j in itr.combinations(nodes, 2)} - dag.skeleton
    for ix, val in enumerate(tqdm(rand_vals)):
        # ADD
        if (val <= add_prob or current_dag.num_arcs == 0) and current_dag.num_arcs != max_arcs:
            i, j = random.choice(list(curr_missing_edges))
            # randomize order
            i, j = (i, j) if extra_rand_vals[ix] < .5 else (j, i)

            if not current_dag.is_upstream_of(j, i):
                updates.append(('add', i, j))
                current_dag.add_arc(i, j, unsafe=True)
                curr_missing_edges.remove(frozenset({i, j}))
            else:
                updates.append(('add', j, i))
                current_dag.add_arc(j, i, unsafe=True)
                curr_missing_edges.remove(frozenset({j, i}))

        # REVERSE
        elif val <= add_prob + reverse_prob:
            # can only reverse i-> to j->i if none of children of i are ancestors of j
            while True:
                i, j = random.choice(list(current_dag.arcs))
                if not any(current_dag.is_upstream_of(c, j) for c in current_dag.children_of(i)):
                    updates.append(('reverse', i, j))
                    current_dag.reverse_arc(i, j, unsafe=True)
                    break
        # REMOVE
        else:
            i, j = random.choice(list(current_dag.arcs))
            current_dag.remove_arc(i, j)
            updates.append(('remove', i, j))
            curr_missing_edges.add(frozenset({i, j}))

        dags.append(current_dag)
        current_dag = current_dag.copy()

    return dags, updates


def add_remove_dag_sampler(dag, nsamples):
    assert nsamples % 2 == 0
    N_all_add = int(1/2 * nsamples)
    N_all_remove = nsamples - int(1/2 * nsamples)
    N_add_remove = nsamples - N_all_add
    dag_set_all_add, updates_all_add = sample_dag_neighborhood(dag, add_prob=1, reverse_prob=0, nsamples=N_all_add)
    dag_set_all_remove, updates_all_remove = sample_dag_neighborhood(dag, add_prob=0, reverse_prob=0, nsamples=N_all_remove)
    return dag_set_all_add + dag_set_all_remove, updates_all_add + updates_all_remove


if __name__ == '__main__':
    d = cd.rand.directed_erdos(500, .5)
    dags, updates = sample_dag_neighborhood_cached(d, reverse_prob=0)
    dags, updates = add_remove_dag_sampler(d, 100)
