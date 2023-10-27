import numpy as np
from numpy import linalg as LA
import scipy

from scipy import optimize
from scipy.optimize import Bounds

def prod_func(x):
    theta = x[0]
    psi = np.array([np.cos(theta), np.sin(theta)])
    A = np.array([[1., 2.], [2., 3.]]); B = np.array([[1., -2.], [-2., 0.]])
    avg_A = np.dot(psi.T, np.dot(A, psi))
    avg_B = np.dot(psi.T, np.dot(B, psi))
    #print(avg_A, avg_B)
    dA = A - avg_A; dB = B - avg_B
    dA_sq = np.dot(dA.T, dA); dB_sq = np.dot(dB.T, dB)
    avg_dA_sq = np.dot(psi.T, np.dot(dA_sq, psi))
    avg_dB_sq = np.dot(psi.T, np.dot(dB_sq, psi))
    prod = avg_dA_sq * avg_dB_sq
    return - prod

theta0 = [0.]
bounds = Bounds([0], [np.pi/2.])
res = optimize.minimize(prod_func, theta0, method='SLSQP', tol=1e-6, bounds=bounds)

print(res)

