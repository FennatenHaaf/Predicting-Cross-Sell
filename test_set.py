# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:06:29 2021

@author: matth
"""
import HMM_eff as ht
import numpy as np 
import pandas as pd

df_per_time = [] 

name_columns = ['p1','p2','p3','var1','var2','var3d1', 'var3d2','var4d1','var4d2', 'ac1','ac2','ac3']

for t in range(0,3):
    #variables for cross-sell
    integers1 = np.random.randint(3, size=(100, 2))
    integers2 = np.random.randint(4, size=(100, 1))
    continu = np.random.uniform(low=0, high=10, size=(100,2))
    binary = np.random.randint(2, size=(100, 4))
    
    #variables for active
    integers3 = np.random.randint(3, size=(100, 2))
    integers4 = np.random.randint(4, size=(100, 1))
    
    matrix = np.concatenate((integers1, integers2, continu, binary, integers3, integers4), axis = 1)
    df = pd.DataFrame(data = matrix, columns = name_columns)
    df_per_time.append(df)
    

name_dep_var_cross_sell = ['p1','p2','p3']
name_covariates = ['var1','var2']
name_dep_var_active = ['ac1', 'ac2', 'ac3']


#Bool: If true: run Model as paas , if False: run a more general HMM
covariates = False

n_segments = 3
#test_active = ht.HMM_eff(df_per_time, name_dep_var_active, covariates = False)
#param_ac, shapes_ac = test_active.EM(n_segments, max_method = 'Nelder-Mead')
#active_value = test_active.active_value(param, shapes, n_segments)


#n_segments = 4
test_cross_sell = ht.HMM_eff(df_per_time, name_dep_var_cross_sell, name_covariates, True)
param_cross, alpha_cross, shapes_cross = test_cross_sell.EM(n_segments, max_method = 'Nelder-Mead')
#cross_sell_target, cross_sell_self, cross_self_total = test_cross_sell.cross_sell_yes_no(param_cross, shapes_cross, n_segments, alpha, active_value):
