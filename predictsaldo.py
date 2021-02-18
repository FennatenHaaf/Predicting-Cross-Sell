# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:57:40 2021

@author: Maudji
"""

import numpy as np 
import pandas as pd

import statsmodels.api as sm

interdir = "./interdata"
data = pd.read_csv(r'C:\Users\Maudji\Documents\GitHub\Predicting-Cross-Sell\interdata\saldopredict.csv')

# set base dummies
y = data['percdiff']


data['geslacht_Man'] = data['geslacht_Man'] + data['geslacht_Mannen']
X = data.drop(columns = [ 'percdiff', 'business_change', 'retail_change','joint_change','retail_change_dummy',  'business_change_dummy', 'joint_change_dummy' ,'saldo_now', 'income_1.0', 'educat4_1.0', 'housetype_1.0', 'lfase_1.0', 'huidigewaarde_klasse_1.0', 'age_bins_(0, 18]', 'geslacht_Man', 'activitystatus_1.0', 'geslacht_Mannen' ])
X['business_retail_joint'] = data['business_change_dummy']* data['retail_change_dummy']*data['joint_change_dummy']
X['business_retail'] = data['business_change_dummy']* data['retail_change_dummy']*(1 - data['joint_change_dummy'])
X['business'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(1 - data['joint_change_dummy'])
X['business_joint'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])
X['joint'] = (1 - data['business_change_dummy'])* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])  
X['retail_joint'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(data['joint_change_dummy'])
X['retail'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(1 -data['joint_change_dummy'])

olsmod = sm.OLS(y, X)
olsres = olsmod.fit()
print(olsres.summary())





