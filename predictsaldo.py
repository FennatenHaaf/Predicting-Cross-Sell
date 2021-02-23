# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:57:40 2021

@author: Maudji
"""

import numpy as np 
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf

interdir = "./interdata"
data = pd.read_csv(r'C:\Users\Maudji\Documents\GitHub\Predicting-Cross-Sell\interdata\saldopredict.csv')

# set base dummies
y = data['percdiff']



data['geslacht_Man'] = data['geslacht_Man'] + data['geslacht_Mannen']
X = data.drop(columns = [ 'percdiff', 'geslacht_Vrouwen','saldo_prev', 'business_change', 'retail_change','joint_change','retail_change_dummy',  'business_change_dummy', 'joint_change_dummy' ,'saldo_now', 'income_1.0', 'educat4_1.0', 'housetype_1.0', 'lfase_1.0', 'huidigewaarde_klasse_1.0', 'age_bins_(0, 18]', 'geslacht_Man', 'activitystatus_1.0', 'geslacht_Mannen' ])
X['business_retail_joint'] = data['business_change_dummy']* data['retail_change_dummy']*data['joint_change_dummy']
X['business_retail'] = data['business_change_dummy']* data['retail_change_dummy']*(1 - data['joint_change_dummy'])
X['business'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(1 - data['joint_change_dummy'])
X['business_joint'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])
X['joint'] = (1 - data['business_change_dummy'])* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])  
X['retail_joint'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(data['joint_change_dummy'])
X['retail'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(1 -data['joint_change_dummy'])
X['constant'] = [1]* X.shape[0]

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()


def backwardel(olsres, X, y):
    r2adjusted = []   # Will store R Square adjusted value for each loop
    r2 = [] 
    
    current_X = X
    max_p = 1
    
    while (max_p >= 0.05):
        p_val = list(olsres.pvalues)
        r2adjusted.append(olsres.rsquared_adj)
        r2.append(olsres.rsquared)
        
        max_p = max(p_val)
        max_p_index = p_val.index(max_p)
        print(current_X.columns[max_p_index], " is dropped with p-val", max_p)
        current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
    
        olsres = sm.OLS(y, current_X).fit()
    
    
    return current_X, olsres, r2adjusted, r2

X_final, ols_final, r2adjusted, r2 = backwardel(olsres, X, y)
print(ols_final.summary())
ypred = ols_final.predict(X_final)
saldo_verschil = np.exp(ypred.array._ndarray) 
new_saldo = data['saldo_now'] + saldo_verschil





