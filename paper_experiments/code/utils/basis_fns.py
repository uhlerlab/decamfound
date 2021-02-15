
import numpy as np

def trunctated_quadratic_trend(x, cutoff=9):
	return np.minimum(x ** 2, cutoff)

def sin_trend(x):
	return np.sin(np.pi * x)

def linear_trend(x):
	return x
