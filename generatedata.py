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
import pandas as pd

def getGeneratedData():
    states = np.array([0, 1])
    p_init = np.array([0.5, 0.5])
    p_trans = np.array([[0.9, 0.1], [0.2,0.8]])
    T = 2
    n_customers = 500
    n_products = 2
    n_segments = 2
    n_categories = 2
    gamma0 = np.array([[0.2, 0.5, 2]])
    gamma_sr_0 = np.array([[0.3, 0.5]])
    gamma_sr_t = np.array([[-0.3, 1.2]])
    #beta = np.array([ [[-0.5],[0.3],[0.4],[-0.4]] , [[0.4],[-0.3],[-0.5],[0.3]] ])
    beta = np.array([ [[-0.5],[0.3]], [[0.4],[-0.4]] ])
    fixed_random_seed = np.random.RandomState(978391)
    
    
    def makecovariates (n_customers, T, n_categories):
        cov = np.zeros((T, n_customers, n_categories+1))
        for i in range(0,n_customers):
            cov[:,i,0] = np.ones(T)
            for c in range(1,n_categories+1):
                cov[:,i,c] = fixed_random_seed.uniform(low=0, high=10, size = (T))
              
        return cov
    
    cov = makecovariates(n_customers, T, n_categories)
    
    def initialstate(cov, n, gamma0):
        prob = np.zeros(n_segments-1)
        for s in range(0,n_segments-1):
            prob[s] = math.exp(np.matmul(gamma0[s,:],cov[0,n,:]))
            
        if (prob > 1):
            initialstate = 0
        else:
            initialstate = 1
        return initialstate
    
    def nextstate(gamma_sr_0, gamma_sr_t, n, cov, prevstate, t):
        prob = math.exp(gamma_sr_0[0,prevstate] + np.matmul(gamma_sr_t[0,:], np.transpose(cov[t,n,1:3]) ))
        
        if (prob > 1):
            nextstate = 0
        else:
            nextstate = 1
        return nextstate   
        
    def markov_chain(n, T, cov, gamma0, gamma_sr_t, gamma_sr_0):
     
        states = np.zeros(T)
        states[0] = initialstate(cov, n, gamma0)
        t =1
        while (t<T):
            states[t] = nextstate(gamma_sr_0, gamma_sr_t, n, cov, int(states[t-1]), t)
            t = t+1
        return states
    
    state = markov_chain(1, T, cov, gamma0, gamma_sr_t, gamma_sr_0)
    
    
    def owndata(beta, n_products, s):
        prob_own = np.zeros(n_products)
        own = np.zeros(n_products)
        for j in range(0, n_products):
            if s == 0:
                prob_own[j] = math.exp(beta[s, j, 0] + beta[1, j, 0])  / (1 + math.exp(beta[s, j, 0] + beta[1, j, 0]))  +  fixed_random_seed.uniform(low=-0.15, high=0.15)
            else:
                prob_own[j] = math.exp(beta[s, j, 0]) / (1 + math.exp(beta[s, j, 0]))  +  fixed_random_seed.uniform(low=-0.15, high=0.15)

            if (prob_own[j] > 0.5):
                own[j] = 1
            else:
                own[j] = 0
        return own
    
    
    def outputdata (beta, n_segments, n_products ,T, cov, gamma0, gamma_sr_0, gamma_sr_t):
        data = np.zeros((T, n_customers,n_products))
        for n in range(0,n_customers):
            state = markov_chain(n, T, cov, gamma0, gamma_sr_0, gamma_sr_t)
            for t in range(0,T):
                currentstate = state[t]
                data[t, n, :] = owndata(beta, n_products, int(currentstate)) 
         
        return data
    
    output = outputdata(beta, n_segments, n_products ,T, cov, gamma0, gamma_sr_0, gamma_sr_t)
    name_columns = ['p1','p2','var1','var2']

    df_gen_data = []
    
    for t in range(0,T):
        matrix = np.concatenate((output[t,:,:], cov[t,:,1:3]), axis = 1)
        df = pd.DataFrame(data = matrix, columns = name_columns )
        df_gen_data.append(df)

    return df_gen_data
    
 



