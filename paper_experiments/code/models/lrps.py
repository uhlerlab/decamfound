import numpy as np
from numpy.linalg import pinv


def update_A():
    X1 = mu * (Shat - Lhat) - C - Uhat


def updateAlpha():
    pass


def updateL():
    pass


def updateS():
    pass


def updateShatLhatUhat():
    pass


def updateU():
    pass


def update_parameters(L, S, U, mu, Uhat, Shat, Lhat, ck, eta):
    ck_new = 1/mu * np.sum((Uhat - U)**2) + mu * np.sum((Shat - Lhat - S + L)**2)
    if ck_new < eta * ck:
        pass
        # updateShatLhatUhat
        # updateAlpha
    else:
        Shat = S
        Uhat = U
        Lhat = L

    return None


def fit_lrps(
        cov,
        lambda1,
        lambda2,
        nsamples,
        init=None,
        maxiter=2000,
        mu=0.1,
        tol=1e-5,
        eta=.999,
        progress=False,
        zeros=None,
        max_rank=None
):
    p = cov.shape[0]

    if init is None:
        S = pinv(cov)
        L = S * .01
        A = S - L
        U = mu + (A - S + L)

    for i in range(maxiter):
        L = update_parameters(parameters)











