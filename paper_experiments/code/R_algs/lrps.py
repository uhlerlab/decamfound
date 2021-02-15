import numpy as np
import os
import subprocess
import ipdb

lrps_filename = os.path.join(os.path.dirname(__file__), 'lrps.R')
lrps_cv_filename = os.path.join(os.path.dirname(__file__), 'cross_validate.R')
lrps_path_filename = os.path.join(os.path.dirname(__file__), 'lrps_path.R')


def lrps(corr, lambda1, lambda2, nsamples):
    corr_filename = 'tmp_corr.npy'
    np.save(corr_filename, corr)

    print(lrps_filename)
    subprocess.call([
        'Rscript',
        lrps_filename,
        corr_filename, str(lambda1), str(lambda2), str(nsamples)
    ])
    S_filename = 'tmp_out_S.npy'
    L_filename = 'tmp_out_L.npy'
    A_filename = 'tmp_out_A.npy'
    U_filename = 'tmp_out_U.npy'
    est_S = np.load(S_filename)
    est_L = np.load(L_filename)
    est_A = np.load(A_filename)
    est_U = np.load(U_filename)

    os.remove(corr_filename)
    os.remove(S_filename)
    os.remove(L_filename)
    os.remove(A_filename)
    os.remove(U_filename)
    return est_S, est_L, est_A, est_U


def lrps_cv(samples):
    samples_filename = 'tmp_samples.npy'
    np.save(samples_filename, samples)

    subprocess.call([
        'Rscript',
        lrps_cv_filename,
        samples_filename
    ])
    out_filename = 'tmp_out_S.npy'
    S = np.load(out_filename)

    os.remove(samples_filename) if os.path.exists(samples_filename) else None
    os.remove(out_filename) if os.path.exists(out_filename) else None
    print(S)
    C = np.linalg.inv(S)
    print(C)
    return C


def lrps_path(samples):
    samples_filename = 'tmp_samples.npy'
    np.save(samples_filename, samples)

    subprocess.call([
        'Rscript',
        lrps_path_filename,
        samples_filename
    ])
    filenames = [f for f in os.listdir(".") if f.startswith("tmp_out")]
    S_path = [np.load(f) for f in filenames]

    os.remove(samples_filename) if os.path.exists(samples_filename) else None
    for f in filenames:
        os.remove(f)
    C_path = [np.linalg.inv(S) for S in S_path]
    return C_path
