
import causaldag as cd
import random
import random
import itertools as itr
import numpy as np
from tqdm import tqdm


def parent_set_perturbation(
        node,
        true_node_parents,
        p
):
	true_node_parents = set(true_node_parents)
	wrong_parents = set(list(range(p))) - true_node_parents - {node}
	parent_neighborhoods = []
	for wrong_parent in wrong_parents:
		parent_neighborhoods.append(sorted(list(true_node_parents | {wrong_parent})))
	random.shuffle(parent_neighborhoods)
	return parent_neighborhoods


def parent_set_removal(
        node,
        true_node_parents,
        p
):
	assert len(true_node_parents) > 0
	true_node_parents = set(true_node_parents)
	parent_neighborhoods = []
	for parent in true_node_parents:
		parent_neighborhoods.append(sorted(list(true_node_parents - {parent})))
	random.shuffle(parent_neighborhoods)
	return parent_neighborhoods
	