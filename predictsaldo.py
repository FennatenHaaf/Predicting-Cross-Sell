"""
Method to predict an increase in saldo for Knab Customers

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import numpy as np 
import pandas as pd
import math

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split 


# =============================================================================
# SELECT DATA
# =============================================================================


interdir = "./interdata"
#data = pd.read_csv(r'C:\Users\Maudji\Documents\GitHub\Predicting-Cross-Sell\interdata\saldopredict.csv')
data = pd.read_csv(f'{interdir}/saldopredict.csv')


#data['geslacht_Man'] = data['geslacht_Man'] + data['geslacht_Mannen'] niet meer nodig, is al gefixt in de dataset


# Select our dependent variable
y = data['percdiff']
X = data.drop(columns = ['percdiff'])

              
#Drop some of the variables that we do not use from x
X = X.drop(columns = ['personid','portfolio_change','saldo_prev','business_change','retail_change','joint_change', 
                      'retail_change_dummy', 'business_change_dummy','joint_change_dummy' ,
                      'saldo_now','saldo_prev'])

# Drop the base cases
X = X.drop(columns = ['income_1.0', 'educat4_1.0', 'housetype_1.0', 'lfase_1.0', 'huidigewaarde_klasse_1.0', 
                      'age_bins_(18, 30]', 'geslacht_Man','activitystatus_1.0' ])
# Drop lfase as well
X = X.drop(columns = [ 'lfase_2.0', 'lfase_3.0', 'lfase_4.0', 'lfase_5.0', 'lfase_6.0', 'lfase_7.0', 'lfase_8.0',])


# Create cross-effects dummies
X['business_retail_joint'] = data['business_change_dummy']* data['retail_change_dummy']*data['joint_change_dummy']
X['business_retail'] = data['business_change_dummy']* data['retail_change_dummy']*(1 - data['joint_change_dummy'])
X['business'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(1 - data['joint_change_dummy'])
X['business_joint'] = data['business_change_dummy']* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])
X['joint'] = (1 - data['business_change_dummy'])* (1 - data['retail_change_dummy'])*(data['joint_change_dummy'])  
X['retail_joint'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(data['joint_change_dummy'])
X['retail'] = (1-data['business_change_dummy'])* (data['retail_change_dummy'])*(1 -data['joint_change_dummy'])
X['constant'] = [1]* X.shape[0]





# =============================================================================
# TRAIN MODEL
# =============================================================================

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

olsmod = sm.OLS(y_train, x_train)
olsres = olsmod.fit()


def backwardel(olsres, X_train, y_train):
    r2adjusted = []   # Will store R Square adjusted value for each loop
    r2 = [] 
    mse =[100]
    
    current_X = X_train
    max_p = 1
    new_p = False
    
    while (max_p >= 0.05):
        if (new_p == False):           
            p_val = list(olsres.pvalues)
            
            r2adjusted.append(olsres.rsquared_adj)
            r2.append(olsres.rsquared)
            xx = x_test[list(current_X.columns)] 
            pred = np.mean(np.square(np.array(olsres.predict(xx) - y_test)))
        if (new_p == True):
            max_p = max(p_val)
            max_p_index = p_val.index(max_p)
            temp_current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
            olsres = sm.OLS(y_train, temp_current_X).fit()
            xx = x_test[list(temp_current_X.columns)]
            pred = np.mean(np.square(np.array(olsres.predict(xx) - y_test)))
            
            
        if (pred < min(mse)):
            new_p = False
            mse.append(pred)
            max_p = max(p_val)
            max_p_index = p_val.index(max_p)
            print(current_X.columns[max_p_index], " is dropped with p-val", max_p)
            current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
    
            olsres = sm.OLS(y_train, current_X).fit()
        else:
            p_val.remove(max(p_val))
            new_p = True
            
    
    return current_X, olsres, r2adjusted, r2, mse

X_final, ols_final, r2adjusted, r2, mse= backwardel(olsres, x_train, y_train)
print(ols_final.summary())
#correlation = X_final.corr()

#ypred = ols_final.predict(x_test[list(X_final.columns)])
#mse = np.mean(np.square(np.array((ypred - y_test))))
#saldo_verschil = np.exp(ypred.array._ndarray) 
#new_saldo = data['saldo_now'] + saldo_verschil



              
            

#print(current_X.columns[max_p_index], " is dropped with p-val", max_p)
#current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
    


