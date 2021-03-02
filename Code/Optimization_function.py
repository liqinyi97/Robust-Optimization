#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import scipy.io
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from scipy.stats import chi2



def Markovitz_opt(mean, cov, t):
    a = 2/t
    N = len(mean)
    
    P = matrix(a*cov)
    q = matrix((-1)*mean)
    G = matrix(np.diag([-1.0 for i in range(N)]), tc='d')
    h = matrix(np.array([0.0 for i in range(N)]), tc='d')
    A = matrix(np.array([[1.0 for i in range(N)]]), tc='d')
    b = matrix(np.array([1.0]), tc='d')
    sol = solvers.qp(P,q,G,h,A,b)
    w = sol['x']
    #CE_0 = -sol['primal objective']
    return w


def Markovitz_opt_risktarget(mean, cov, risk_level):
    N = len(mean) 
    X = np.linalg.cholesky(cov)
    
    c = matrix(-mean)
    A = matrix(np.array([[1.0 for i in range(N)]]), tc='d')
    b = matrix(np.array([1.0]), tc='d')
    
    Gl = matrix(np.diag([-1.0 for i in range(N)]), tc='d')
    hl = matrix(np.array([0.0 for i in range(N)]), tc='d')
    Gq = [matrix(np.vstack([np.array([0.0 for i in range(N)]), (-1)*X.T]))]
    hq = [matrix(np.array([risk_level] + [0.0 for i in range(N)]))]
    
    sol = solvers.socp(c, Gl = Gl, hl = hl, Gq = Gq, hq = hq, A = A, b = b)
    w = sol['x']
    #CE_0 = -sol['primal objective']
    return w



def Robust_opt(mean, cov, sigma, t, eta):
    N = len(mean)
    a = 2/t
    kappa = np.sqrt(chi2.ppf(eta, df = N))
    Y = np.linalg.cholesky(sigma)
    
    P_0 = np.hstack([a*cov, np.array([[0.0] for i in range(N)])])
    P_0 = np.vstack([P_0, np.array([0.0 for i in range(N+1)])])
    P = matrix(P_0)
    
    q = matrix(np.append((-1)*mean, [kappa]))
    
    
    A = matrix(np.array([[1.0 for i in range(N)] + [0.0]]), tc='d')
    b = matrix(np.array([1.0]), tc='d')
    
    I = matrix(0.0, (N+1,N+1))
    I[::N+2] = 1.0
    G_1 = np.hstack([(-1)*Y.T, np.array([[0.0] for i in range(N)])])
    G_1 = np.vstack([np.array([0.0 for i in range(N)] + [-1.0]), G_1])
    G = matrix([-I, matrix(G_1)])
    h = matrix((N+1)*[0.0] + (N+1)*[0.0])
    
    dims = {'l': N+1, 'q': [N+1], 's': []}
    
    sol = solvers.coneqp(P, q, G, h, dims, A, b)
    w = sol['x'][:-1]
    #CE_0 = np.dot(mean,w)[0] - 0.5*a*np.dot(np.dot(w.T, cov), w)[0][0]
    
    return w


def Robust_opt_risktarget(mean, cov, sigma, eta, risk_level):
    N = len(mean)
    kappa = np.sqrt(chi2.ppf(eta, df = N))
    X = np.linalg.cholesky(cov)
    Y = np.linalg.cholesky(sigma)
    
    c = matrix(np.append(-mean, [kappa]))
    A = matrix(np.array([[1.0 for i in range(N)] + [0.0]]), tc='d')
    b = matrix(np.array([1.0]), tc='d')
    
    Gl = matrix(np.diag([-1.0 for i in range(N+1)]), tc='d')
    hl = matrix(np.array([0.0 for i in range(N+1)]), tc='d')
    G_0 = np.hstack([(-1)*X.T, np.array([[0.0] for i in range(N)])])
    G_0 = np.vstack([np.array([0.0 for i in range(N+1)]), G_0])
    Gq = [matrix(G_0)]
    hq = [matrix(np.array([risk_level] + [0.0 for i in range(N)]))]
    G_1 = np.hstack([(-1)*Y.T, np.array([[0.0] for i in range(N)])])
    G_1 = np.vstack([np.array([0.0 for i in range(N)] + [-1.0]), G_1])
    Gq += [matrix(G_1)]
    hq += [matrix(np.array([0.0 for i in range(N+1)]))]
    
    sol = solvers.socp(c, Gl = Gl, hl = hl, Gq = Gq, hq = hq, A = A, b = b)
    w = sol['x'][:-1]
    
    return w
    