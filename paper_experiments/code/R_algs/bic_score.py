import numpy as np
import os
import subprocess


def bic_score_R(dag, samples):
    amat, _ = dag.to_amat()
    samples_filename = 'tmp_samples.npy'
    np.save(samples_filename, samples)
    amat_filename = 'tmp_amat.npy'
    np.save(amat_filename, amat.astype(float))

    subprocess.call(['Rscript', 'paper_experiments/code/R_algs/bic_score.R', samples_filename, amat_filename])
    output_filename = 'tmp_output.npy'
    score = np.load(output_filename)

    os.remove(samples_filename)
    os.remove(amat_filename)
    os.remove(output_filename)
    return score


def bic_score_bnlearn_R(dag, samples):
    amat, o = dag.to_amat()
    samples_filename = 'tmp_samples.npy'
    np.save(samples_filename, samples)
    amat_filename = 'tmp_amat.npy'
    np.save(amat_filename, amat.astype(float))

    subprocess.call(['Rscript', 'paper_experiments/code/R_algs/bic_score_bnlearn.R', samples_filename, amat_filename])
    output_filename = 'tmp_output.npy'
    score = np.load(output_filename)

    os.remove(samples_filename)
    os.remove(amat_filename)
    os.remove(output_filename)
    return score
