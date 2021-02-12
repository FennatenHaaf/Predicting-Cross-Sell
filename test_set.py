# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:06:29 2021

@author: matth
"""
import HMM_eff as ht
import numpy as np 
import pandas as pd

df_per_time = [] 

name_columns = ['p1','p2','p3','var1','var2','var3d1', 'var3d2','var4d1','var4d2']

for t in range(0,3):
    integers1 = np.random.randint(3, size=(100, 2))
    integers2 = np.random.randint(4, size=(100, 1))
    
    continu = np.random.uniform(low=0, high=10, size=(100,2))
    binary = np.random.randint(2, size=(100, 4))
    matrix = np.concatenate((integers1, integers2, continu, binary), axis = 1)
    df = pd.DataFrame(data = matrix, columns = name_columns)
    df_per_time.append(df)
    

name_dep_var = ['p1','p2','p3']
#name_covariates = ['var1','var2','var3d1', 'var3d2','var4d1','var4d2']
name_covariates = ['var1','var2']

#Bool: If true: run Model as paas , if False: run a more general HMM
covariates = False

test = ht.HMM_eff(df_per_time, name_dep_var, name_covariates, covariates)
n_segments = 4


gamma_0, gamma_sr_0, gamma_sk_t, beta = test.EM(n_segments, max_method = 'Nelder-Mead')


