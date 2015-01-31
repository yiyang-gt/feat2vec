
"""
Marginalized Linear Denoising Autoencoders with Structured Dropout noise

Author: Yi Yang
Email: yangyiycc@gmail.com
"""

from __future__ import division
import numpy as np
import scipy.sparse as sp

def mldae(xx, pivots, lam=1.0):
    xx = xx.transpose() # d x n input matrix
    xfreq = xx[pivots,:]
    P = xfreq * xx.transpose()
    normvec = np.squeeze(np.asarray(xx.sum(1)))
    d = len(normvec)
    normvec = normvec + lam*np.ones(d)
    normvec = np.ones(d) / normvec
    Q = sp.spdiags([normvec],[0],d,d)
    W = P * Q
    return W.transpose() # d x h projection matrix

