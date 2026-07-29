"""
# In this file most of the fit functions for all other routines are located, e.g. gaussians, linear functions etc.
#
# Created by Lukas Stock, 13.04.2026
"""

import numpy as np

def gaussian(x, A, x0, sigma):
    return A * np.exp(- np.power((x - x0)/sigma, 2.))

#modified gaussian
def ModGauss(x, amplitude, center, sigma, p):
    return amplitude * np.exp( -0.5 * (np.power(np.abs(center - x), p)/np.power(sigma, p)))

#modified double gaussian
def ModDoubleGauss(x, amp1, cent1, sig1, p1, amp2, cent2, sig2, p2):
    return ModGauss(x, amp1, cent1, sig1, p1) + ModGauss(x, amp2, cent2, sig2, p2)

def linear(x, a, b):
    return a*x + b

def constant(x, b):
    return b

def gaussianAndConstant(x, A, x0, sigma, b):
    return gaussian(x, A, x0, sigma) + constant(x, b)

def doublegaussianAndConstant(x, A1, x1, sigma1, A2, x2, sigma2, b):
    return gaussian(x, A1, x1, sigma1) + gaussian(x, A2, x2, sigma2) + constant(x, b)
