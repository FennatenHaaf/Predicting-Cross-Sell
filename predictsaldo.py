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
    
    
    def __init__(self, saldo_data = None, df_time_series = None, interdir = "./interdata", 
                 cross_sell_types = None, drop_variables = None, base_variables = None):
        """
        Parameters
        ----------
        saldo_data : dataframe
            dataframe consisting of all the data that is needed to estimate the parameters for the prediction
        df_time_series : list of dataframes
            dataframes on which saldo_data is based, needed for the prediction itself.
        interdir : string
            direction of the map where inter results are stored
        cross_sell_types : list of strings
            list consisting of strings that represent the products, thus the potential cross sells
        drop_variables : list of strings
            list consisting of variables that can be dropped from saldo_data
        base_variables : list of strings
            list consisting of the dummyvariables that are the base cases
            
        """
        """function for the initialisation of an predict saldo object"""
        
        
        self.interdir = interdir
        
        # initialise dataframes
        if saldo_data == None:        
            self.saldo_data = pd.read_csv(f'{interdir}/saldopredict.csv')
        else: 
            self.saldo_data = saldo_data
            
        if df_time_series == None:
            name = "final_df"
            self.df_time_series = [pd.read_csv(f"{interdir}/{name}_2018Q1.csv"),
                            pd.read_csv(f"{interdir}/{name}_2018Q2.csv"),
                            pd.read_csv(f"{interdir}/{name}_2018Q3.csv"),
                            pd.read_csv(f"{interdir}/{name}_2018Q4.csv"),
                            pd.read_csv(f"{interdir}/{name}_2019Q1.csv"),
                            pd.read_csv(f"{interdir}/{name}_2019Q2.csv"),
                            pd.read_csv(f"{interdir}/{name}_2019Q3.csv"),
                            pd.read_csv(f"{interdir}/{name}_2019Q4.csv"),
                            pd.read_csv(f"{interdir}/{name}_2020Q1.csv"),
                            pd.read_csv(f"{interdir}/{name}_2020Q2.csv"),
                            pd.read_csv(f"{interdir}/{name}_2020Q3.csv"),
                            pd.read_csv(f"{interdir}/{name}_2020Q4.csv")]
        else:
            self.df_times_series = df_time_series
            
        if cross_sell_types == None:        
            self.cross_sell_types = ["business",
                                     "retail",
                                     "joint",
                                     "accountoverlay"]
        else: 
            self.cross_sell_types = cross_sell_types
            
            
        # Select our dependent variable
        y = self.saldo_data['percdiff']
        X = self.saldo_data.drop(columns = ['percdiff'])

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
        X['business_retail_joint'] = saldo_data['business_change_dummy']* saldo_data['retail_change_dummy']*saldo_data['joint_change_dummy']
        X['business_retail'] = saldo_data['business_change_dummy']* saldo_data['retail_change_dummy']*(1 - saldo_data['joint_change_dummy'])
        X['business'] = saldo_data['business_change_dummy']* (1 - saldo_data['retail_change_dummy'])*(1 - saldo_data['joint_change_dummy'])
        X['business_joint'] = saldo_data['business_change_dummy']* (1 - saldo_data['retail_change_dummy'])*(saldo_data['joint_change_dummy'])
        X['joint'] = (1 - saldo_data['business_change_dummy'])* (1 - saldo_data['retail_change_dummy'])*(saldo_data['joint_change_dummy'])  
        X['retail_joint'] = (1-saldo_data['business_change_dummy'])* (saldo_data['retail_change_dummy'])*(saldo_data['joint_change_dummy'])
        X['retail'] = (1-saldo_data['business_change_dummy'])* (saldo_data['retail_change_dummy'])*(1 -saldo_data['joint_change_dummy'])
        X['constant'] = [1]* X.shape[0]
        
        self.y = y
        self.X = X


    def train_predict(self, test_set_prop = 0.2, random_state = 0, p_bound = 0.05):
        """
        Parameters
        ----------
        test_set_prop : float
            proportion of the data that is used for testing
        random_state : int
            seed for taking a random training/test set
        p_bound : float
            bound that indicates whether a variable is significant
        Returns
        -------
        X_var_final : list
            list of variables that turn out to be significant for predicing the extra saldo
        ols_final : ols object
            object of the final ols of the backward elimination for predicting the extra saldo
        r2adjusted  : list
            list of adjusted r2's of the backward elimination
        r2 : list
            list of r2's of the backward elimination
        mse : list
            list of mse's of the backward elimination
        """
        """function for training the saldo prediction model"""
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_set_prop, random_state = random_state) 

        olsmod = sm.OLS(y_train, X_train)
        olsres = olsmod.fit()

        X_var_final, ols_final, r2adjusted, r2, mse = self.backwardel(olsres, X_train, X_test, y_train, y_test, p_bound)
        print(ols_final.summary())
        
        return X_var_final, ols_final, r2adjusted, r2, mse 

    
    def backwardel(self, olsres, X_train, X_test, y_train, y_test, p_bound):
        """
        Parameters
        ----------
        olsres : ols object
            ols object where all variables are included
        X_train : dataframe
            training set of the independent variables 
        X_test : dataframe
            test set of the independent variables 
        y_train : dataframe
            training set of the dependent variables 
        y_test : dataframe
            test set of the dependent variables 
        Returns
        -------
        X_var_final : list
            list of variables that turn out to be significant for predicing the extra saldo
        ols_final : ols object
            object of the final ols of the backward elimination for predicting the extra saldo
        r2adjusted  : list
            list of adjusted r2's of the backward elimination
        r2 : list
            list of r2's of the backward elimination
        mse : list
            list of mse's of the backward elimination
        """
        """function for the backward elimination of the saldo predicition model"""
        
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

        return current_X_var, olsres, r2adjusted, r2, mse
    
    
        def get_fitted_values(self, cross_sell_types, cross_sell_yes_no, df_ts, test_set_prop, random_state, p_bound):
            """
            Parameters
            ----------
            cross_sell_types : list of strings
                list consisting of strings that represent the products, thus the potential cross sells
            cross_sell_yes_no : 2D array
                array indicating whether customers (rows) are eligible for the cross sell of certain products (columns)
            df_ts : dataframe
                dataframe for predicting the extra saldo 
            test_set_prop : float
                proportion of the data that is used for testing
            random_state : int
                seed for taking a random training/test set
            p_bound : float
                bound that indicates whether a variable is significant
                    
            Returns
            -------
            fitted_values : 1D array
                array with the fitted values of the saldo prediction model
            X_var_final : list
                list of variables that turn out to be significant for predicing the extra saldo
            ols_final : ols object
                object of the final ols of the backward elimination for predicting the extra saldo
            """
            """function for calculating the fitted values of the saldo prediction model"""
        
            
            # train model to get parameters
            X_var_final, ols_final, r2adjusted, r2, mse = self.train_predict(test_set_prop, random_state, p_bound)
            
            df_cross_sell = pd.DataFrame(data = cross_sell_yes_no, columns = cross_sell_types)

            # create cross-effects dummies
            df_ts['business_retail_joint'] = df_cross_sell['business']* df_cross_sell['retail']*df_cross_sell['joint']
            df_ts['business_retail'] = df_cross_sell['business']* df_cross_sell['retail']*(1 - df_cross_sell['joint'])
            df_ts['business'] = df_cross_sell['business']* (1 - df_cross_sell['retail'])*(1 - df_cross_sell['joint'])
            df_ts['business_joint'] = df_cross_sell['business']* (1 - df_cross_sell['retail'])*(df_cross_sell['joint'])
            df_ts['joint'] = (1 - df_cross_sell['business'])* (1 - df_cross_sell['retail'])*(df_cross_sell['joint'])  
            df_ts['retail_joint'] = (1-df_cross_sell['business'])* (df_cross_sell['retail'])*(df_cross_sell['joint'])
            df_ts['retail'] = (1-df_cross_sell['business'])* (df_cross_sell['retail'])*(1 -df_cross_sell['joint'])
            df_ts['constant'] = [1]* df_ts.shape[0]
            
            # use significant variables and corresponding parameters
            df_ts_final = df_ts[X_var_final]
            beta = ols_final.params
            
            # calculate fitted values
            fitted_values = self.df_ts_final.dot(beta)
            
            return fitted_values, X_var_final, ols_final

        def fitted_values_to_saldo(self, minimum, fitted_values, df):
            """
            Parameters
            ----------
            minimum : float
                minimum of all the saldos of all customers over all the time
            fitted_values : 1D array
                array with the fitted values of the saldo prediction model
            df : dataframe
                dataframe for predicting the extra saldo 
            Returns
            -------
            extra_saldo : 1D array
                array consisting of all the extra saldos on the account balances when cross_sells are done   
            """
            """function that derives the actual extra saldo from the fitted values"""
        

            prev_saldo = self.df[[["business","retail",
                                            "joint"]].sum(axis=1)]
            
            extra_saldo = math.exp(fitted_values) * (prev_saldo + minimum) - minimum
            
            return extra_saldo
        
        def get_extra_saldo(self, cross_sell_yes_no, t, minimum, fin_segment = None, test_set_prop = 0.2, random_state = 0, p_bound = 0.05):
            """
            Parameters
            ----------
            cross_sell_yes_no : 2D array
                array indicating whether customers (rows) are eligible for the cross sell of certain products (columns)
            t : int
                time for which one wants to calculate extra saldos
            minimum : float
                minimum of all the saldos of all customers over all the time
            fin_segment : string
                segments for which one wants to predict the extra saldos
            test_set_prop : float
                proportion of the data that is used for testing
            random_state : int
                seed for taking a random training/test set
            p_bound : float
                bound that indicates whether a variable is significant  
            Returns
            -------
            extra_saldo : 1D array
                array consisting of all the extra saldos on the account balances when cross_sells are done   
            X_var_final : list
                list of variables that turn out to be significant for predicing the extra saldo
            ols_final : ols object
                object of the final ols of the backward elimination for predicting the extra saldo
            """
            """function that predicts the extra saldo on the account balances when cross sells are done"""
        
            # initialise the dataframes
            if fin_segment == None:
                df_ts = self.df_time_series[t]
            else: 
                df_ts = self.df_time_series[t]     
                df_ts = self.df_ts[df_ts['finergy_tp'] == fin_segment]

        
            fitted_values, X_var_final, ols_final = self.get_fitted_values(self.cross_sell_types, cross_sell_yes_no, df_ts, 
                                                                                             test_set_prop, random_state, p_bound)
            
            extra_saldo = self.fitted_values_to_saldo(fitted_values, df_ts, minimum)

            return extra_saldo, X_var_final, ols_final



