# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:37:37 2021

@author: Maudji
"""

import numpy as np
from scipy.stats import multinomial
from typing import List
from scipy.stats import norm
import math


states = np.array([0, 1])
p_init = np.array([0.5, 0.5])
p_trans = np.array([[0.9, 0.1], [0.2,0.8]])
T = 2
N = 10
J =2
S = 2
beta = np.array([[1, 1],[2,2]])
C = 2
gamma0= np.array([[1,1,1],[0,0,0]])
gamma_t = np.array([[[1,1,1],[2,1,1]],[[1,1,1],[2,1,1]]])

def makecovariates (N, T, C):
    cov = np.zeros((N, C+1, T))
    for i in range(0,N):
        cov[i,0,:] = np.ones(T)
        for c in range(1,C+1):
            cov[i,c,:] = np.random.random((1,T))
          
    return cov

cov = makecovariates(N, T, C)

def initialstate(cov, N, gamma0):
    prob = np.zeros(S-1)
    for s in range(0,S-1):
        prob[s] = math.exp(np.matmul(gamma0[s,:],cov[N,:,0]))
    if (prob > 1):
        initialstate = 0
    else:
        initialstate = 1
    return initialstate

def nextstate(gamma_t, n, cov, prevstate, t):
    prob = math.exp(np.matmul(gamma_t[prevstate,0,:],cov[n,:,t]))
    if (prob >1):
        nextstate = 0
    else:
        nextstate =1
    return nextstate   
    
def markov_chain(n, T, cov, gamma0, gamma_t):
 
    states = np.zeros(T)
    states[0] = initialstate(cov, n, gamma0)
    t =1
    while (t<T):
        states[t] = nextstate(gamma_t, n, cov, int(states[t-1]), t)
        t = t+1
    return states

state = markov_chain(1, T, cov, gamma0, gamma_t)



def owndata(beta, J, s):
    prob_own = np.zeros(J)
    own = np.zeros(J)
    for j in range(0, J):
        prob_own[j] = math.exp(sum(beta[s])) / (1+ math.exp(sum(beta[s]))) + norm.rvs(0, 1) 
        if (prob_own[j] > 0.5):
            own[j] = 1
        else:
            own[j] = 0
    return own

def outputdata (beta, S, J, T, cov, gamma0, gamma_t):
    data = np.zeros((N,J,T))
    for n in range(0,N):
        state = markov_chain(n, T, cov, gamma0, gamma_t)
        for t in range(0,T):
            currentstate = state[t]
            data[n, :, t] = owndata(beta, J, int(currentstate)) 
     
    return data

output = outputdata(beta, S, J,T, cov, gamma0, gamma_t)

    
 



