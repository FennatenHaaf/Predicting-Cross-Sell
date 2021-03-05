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

class predict_saldo: 
    
    
    def __init__(self, predict_data = None, interdir = "./interdata", drop_variables = None, base_variables = None):

        self.interdir = interdir
        
        if predict_data == None:        
            self.predict_data = pd.read_csv(f'{interdir}/saldopredict.csv')
        else: 
            self.predict_data = predict_data

        # Select our dependent variable
        y = self.predict_data['percdiff']
        X = self.predict_data.drop(columns = ['percdiff'])

        if drop_variables == None:
            #Drop some of the variables that we do not use from x
            X = X.drop(columns = ['personid','portfolio_change','saldo_prev',
                                  'business_change','retail_change','joint_change',
                                  'accountoverlay_change','retail_change_dummy',
                                  'business_change_dummy','joint_change_dummy' ,
                                  'accountoverlay_change_dummy',
                                  'saldo_now','saldo_prev'])
            # Drop lfase as well
            X = X.drop(columns = [ 'lfase_2.0', 'lfase_3.0', 'lfase_4.0', 'lfase_5.0', 
                                  'lfase_6.0', 'lfase_7.0', 'lfase_8.0'])
        else: 
            X = X.drop(columns = drop_variables)
        
        
        if base_variables == None:
            # Drop the base cases
            X = X.drop(columns = ['income_1.0', 'educat4_1.0', 'housetype_1.0', 'lfase_1.0', 
                              'huidigewaarde_klasse_1.0', 'age_bins_(18, 30]', 
                              'geslacht_Man','activitystatus_1.0' ])
        else: 
            X = X.drop(columns = base_variables)

        
        # Create cross-effects dummies
        X['business_retail_joint'] = predict_data['business_change_dummy']* predict_data['retail_change_dummy']*predict_data['joint_change_dummy']
        X['business_retail'] = predict_data['business_change_dummy']* predict_data['retail_change_dummy']*(1 - predict_data['joint_change_dummy'])
        X['business'] = predict_data['business_change_dummy']* (1 - predict_data['retail_change_dummy'])*(1 - predict_data['joint_change_dummy'])
        X['business_joint'] = predict_data['business_change_dummy']* (1 - predict_data['retail_change_dummy'])*(predict_data['joint_change_dummy'])
        X['joint'] = (1 - predict_data['business_change_dummy'])* (1 - predict_data['retail_change_dummy'])*(predict_data['joint_change_dummy'])  
        X['retail_joint'] = (1-predict_data['business_change_dummy'])* (predict_data['retail_change_dummy'])*(predict_data['joint_change_dummy'])
        X['retail'] = (1-predict_data['business_change_dummy'])* (predict_data['retail_change_dummy'])*(1 -predict_data['joint_change_dummy'])
        X['constant'] = [1]* X.shape[0]
        
        self.y = y
        self.X = X


    def train_predict(self, test_set_prop = 0.2, random_state = 0, p_bound = 0.05):
        # =============================================================================
        # TRAIN MODEL
        # =============================================================================

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_set_prop, random_state = random_state) 

        olsmod = sm.OLS(y_train, X_train)
        olsres = olsmod.fit()

        X_final, X_var_final, ols_final, r2adjusted, r2, mse = self.backwardel(olsres, X_train, X_test, y_train, y_test, p_bound)
        print(ols_final.summary())
        
        return X_final, X_var_final, ols_final, r2adjusted, r2, mse 

        
        
    def backwardel(self, olsres, X_train, X_test, y_train, y_test, p_bound):
        r2adjusted = []   # Will store R Square adjusted value for each loop
        r2 = [] 
        mse =[100]
        
        current_X = X_train
        max_p = 1
        new_p = False
        
        while (max_p >= p_bound):
            
            if (new_p == False):           
                p_val = list(olsres.pvalues)
                
                r2adjusted.append(olsres.rsquared_adj)
                r2.append(olsres.rsquared)
                xx = X_test[list(current_X.columns)] 
                pred = np.mean(np.square(np.array(olsres.predict(xx) - y_test)))
            else:
                max_p = max(p_val)
                max_p_index = p_val.index(max_p)
                temp_current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
                olsres = sm.OLS(y_train, temp_current_X).fit()
                xx = X_test[list(temp_current_X.columns)]
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
                
            current_X_var_arr = current_X.columns.values
            current_X_var = list(current_X_var_arr)

        return current_X, current_X_var, olsres, r2adjusted, r2, mse
    
    # hoe is predict data gemaakt? welke tijdsperiode?
        def get_fitted_values(self, cross_sell_yes_no, segment, t = 11, test_set_prop = 0.2, random_state = 0, p_bound = 0.05):
            self.predict_data_seg = pd.read_csv(f'{interdir}/saldopredict_fin{segment}.csv')

            
                train_predict(self, test_set_prop, random_state, p_bound)




        def fitted_values_to_saldo():
            
            
            
            
            
        #correlation = X_final.corr()
        
        #ypred = ols_final.predict(x_test[list(X_final.columns)])
        #mse = np.mean(np.square(np.array((ypred - y_test))))
        #saldo_verschil = np.exp(ypred.array._ndarray) 
        #new_saldo = data['saldo_now'] + saldo_verschil



             
