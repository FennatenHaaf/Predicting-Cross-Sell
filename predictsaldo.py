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

import utils

# =============================================================================
# SELECT DATA
# =============================================================================

class predict_saldo: 
    
    
    def __init__(self, saldo_data = None, df_time_series = None, interdir = "./interdata", 
                 outdir = "./output", cross_sell_types = None, drop_variables = None, 
                 base_variables = None):
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
        self.outdir = outdir
        
        #initialise dataframe for training the model
        if isinstance(saldo_data, type(None)):        
            self.saldo_data = pd.read_csv(f'{interdir}/saldopredict.csv')
        else: 
            self.saldo_data = saldo_data
            
        #initialise dataframe for predicting actual sald
        if isinstance(df_time_series, type(None)):
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
            self.df_time_series = df_time_series
        
        #get names of cross-sell types, used variables within the dataframes
        if isinstance(cross_sell_types, type(None)):        
            self.cross_sell_types = ["business",
                                     "retail",
                                     "joint",
                                     "accountoverlay"]
        else: 
            self.cross_sell_types = cross_sell_types
            
            
        # Select our dependent variable
        y = self.saldo_data['percdiff']
        X = self.saldo_data.drop(columns = ['percdiff'])

        #Drop some of the variables that we do not use from x
        if isinstance(drop_variables, type(None)):
            X = X.drop(columns = ['personid','portfolio_change','saldo_prev',
                                  'business_change','retail_change','joint_change',
                                  'accountoverlay_change','retail_change_dummy',
                                  'business_change_dummy','joint_change_dummy' ,
                                  #'accountoverlay_change_dummy',
                                  'saldo_now','saldo_prev'])
            # Drop lfase as well
            X = X.drop(columns = [ 'lfase_2.0', 'lfase_3.0', 'lfase_4.0', 'lfase_5.0', 
                                  'lfase_6.0', 'lfase_7.0', 'lfase_8.0'])
        else: 
            X = X.drop(columns = drop_variables)
        
        #Drop the base cases
        if isinstance(base_variables, type(None)):
            X = X.drop(columns = ['income_1.0', 'educat4_1.0', 'housetype_1.0', 'lfase_1.0', 'hh_size_1.0',
                              'huidigewaarde_klasse_1.0','age_bins_(0, 18]','age_bins_(18, 30]', 
                              'geslacht_Man', 'geslacht_Man(nen) en vrouw(en)',
                              'activitystatus_1.0' ])
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
        
        #set datasets for training purposes
        self.y = y
        self.X = X


    def train_predict(self, test_set_prop = 0.2, random_state = 0, p_bound = 0.05,
                      outname ="predictsaldo_output"):
        """
        Parameters
        ----------
        test_set_prop : float
            proportion of the data that is used for testing
        random_state : int
            seed for taking a random training/test set
        p_bound : float
            bound that indicates whether a variable is significant
        outname : string
            name for the file in which we can save the trained model 
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
        
        #split the training dataset in a training and testset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size = test_set_prop, random_state = random_state) 

        #train the model by performing backward elimination
        X_var_final, ols_final, r2adjusted, r2, mse = self.backwardel(X_train, X_test, y_train, y_test, p_bound)
        #print summary of final ols model
        print(ols_final.summary())
        
        
        print(f"saving OLS results to {outname}.csv")
         
        #safe the final ols_model to a csv file
        df = pd.DataFrame(columns = [ "coef", "stderr", "pval", 'coefprint'])  
        df["coef"]  = ols_final.params
        df["stderr"] = ols_final.bse
        df["pval"] = ols_final.pvalues
        df = df.reset_index()
        df = df.rename(columns={"index": "covariate"})
        
        # Add the stars to the string so that it becomes easier to copy 
        for i in range(0,len(df)):
           coeff = round(df.loc[i, "coef"],3)
           std =  round(df.loc[i, "stderr"],3)
    
           if (df.loc[i,"pval"] < 0.05):
               if (df.loc[i,"pval"] < 0.01):
                   if (df.loc[i,"pval"] < 0.001):
                       df.loc[i, "coefprint"] = f"{coeff}*** ({std})"
                   else:
                       df.loc[i, "coefprint"] = f"{coeff}** ({std})"
               else:
                  df.loc[i, "coefprint"] = f"{coeff}* ({std})"
           else:
               df.loc[i, "coefprint"] = f"{coeff} ({std})"
        
    
        utils.save_df_to_csv(df, self.outdir, outname, add_time = False )  
        
        return X_var_final, ols_final, r2adjusted, r2, mse 

    
    def backwardel(self, X_train, X_test, y_train, y_test, p_bound):
        """
        Parameters
        ----------
        X_train : dataframe
            training set of the independent variables 
        X_test : dataframe
            test set of the independent variables 
        y_train : dataframe
            training set of the dependent variables 
        y_test : dataframe
            test set of the dependent variables 
        p_bound : float
            bound for which every parameter has to have a lower p-value in the backward elimination
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
        """function for the backward elimination of the saldo predicting model"""
        
        #initialise list of (adjusted) R2 values
        r2adjusted = []
        r2 = [] 
        
        #initialise list of mse's
        mse =[100]
        
        #initialise current X and the maximum of (not yet) obtained p-values
        current_X = X_train
        max_p = 1

        #set new_p = True, so start with an ols estimation
        new_p = True

        #stay in loop while the max of obtained p-values is still higher than the set bound of p-values
        while (max_p >= p_bound):
              
            #perform new OLS, and calculate performance if variable with highest p-value is removed
            if new_p:
                #perform new OLS
                olsres = sm.OLS(y_train, current_X).fit()
                
                #get p-values
                p_val = list(olsres.pvalues)
                #get highest p-value and it's index
                max_p = max(p_val)
                max_p_index = p_val.index(max_p)
                
                #get temporary training dataset of X's without the variable with the highest p-value
                temp_current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
                #perform new OLS with this temporary dataset
                olsres = sm.OLS(y_train, temp_current_X).fit()
                
                #calculate prediction performance of the temporary training dataset on the test-set
                xx = X_test[list(temp_current_X.columns)]
                pred = np.mean(np.square(np.array(olsres.predict(xx) - y_test)))
            #use old OLS, but look at the list of p-values but without the highest p-values of which removing leads to lower MSE
            else:        
                #get highest p-value and it's index
                max_p = max(p_val)
                max_p_index = p_val.index(max_p)
                
                #get temporary training dataset of X's without the variable with the highest p-values
                temp_current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
                #perform new OLS with this temporary dataset
                olsres = sm.OLS(y_train, temp_current_X).fit()
                
                #calculate prediction performance of the temporary training dataset on the test-set
                xx = X_test[list(temp_current_X.columns)] 
                pred = np.mean(np.square(np.array(olsres.predict(xx) - y_test)))

            #OLS with temporary dataset performs well (MSE does get smaller), remove that variable
            if (pred < min(mse)):
                #add pred to the list of MSE's
                mse.append(pred)
                
                #add (adjusted) R2 values to list of (adjusted) R2 values
                r2adjusted.append(olsres.rsquared_adj)
                r2.append(olsres.rsquared)
                
                #remove variable
                current_X = current_X.drop(current_X.columns[max_p_index], axis=1)
                print(current_X.columns[max_p_index], " is dropped with p-val", max_p)

                
                p_val.remove(max(p_val))
                max_p = max(p_val)

                new_p = True
            # OLS with variable with highest p-value does not perform well, remove the p-value
            else:
                p_val.remove(max(p_val))
                max_p = max(p_val)

                new_p = False
            
            olsres = sm.OLS(y_train, current_X).fit()
            current_X_var_arr = current_X.columns.values
            current_X_var = list(current_X_var_arr)

        return current_X_var, olsres, r2adjusted, r2, mse
    

    def get_fitted_values(self, cross_sell_types, cross_sell_total, cross_sell_self, df_ts, test_set_prop, 
                          random_state, p_bound, X_var_final = None, ols_final = None):
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
    
        
        # PREDICT SALDO WITH ALL CROSS SELLS
        df_cs = pd.DataFrame(data = cross_sell_total, columns = cross_sell_types)        
        df_ts_cs = df_ts.copy()
        
        # create cross-effects dummies
        df_ts_cs['business_retail_joint'] = df_cs['business'].copy()* df_cs['retail'].copy()*df_cs['joint'].copy()
        df_ts_cs['business_retail'] = df_cs['business'].copy()* df_cs['retail'].copy()*(1 - df_cs['joint'].copy())
        df_ts_cs['business'] = df_cs['business'].copy()* (1 - df_cs['retail'].copy())*(1 - df_cs['joint'].copy())
        df_ts_cs['business_joint'] = df_cs['business'].copy()*(1 - df_cs['retail'].copy())*(df_cs['joint'].copy())
        df_ts_cs['joint'] = (1 - df_cs['business'].copy())* (1 - df_cs['retail'].copy())*(df_cs['joint'].copy())  
        df_ts_cs['retail_joint'] = (1-df_cs['business'].copy())* (df_cs['retail'].copy())*(df_cs['joint'].copy())
        df_ts_cs['retail'] = (1-df_cs['business'].copy())* (df_cs['retail'].copy())*(1 -df_cs['joint'].copy())
        df_ts_cs['constant'] = np.ones(len(df_ts_cs))  #[1]* df_ts.shape[0]
        
        # add a few variables which are not standard in the dataset
        df_ts_cs['log_saldoprev'] = df_ts_cs["log_saldototaal"].copy()
        df_ts_cs['prev_retail_dummy'] = df_ts_cs["retail_dummy"].copy()
        df_ts_cs['prev_joint_dummy'] = df_ts_cs["joint_dummy"].copy()
        df_ts_cs['prev_business_dummy'] = df_ts_cs["business_dummy"].copy()
        
        
        
        
        # PREDICT SALDO WITHOUT 'TARGET' CROSS SELLS
        df_no_targ = pd.DataFrame(data = cross_sell_self, columns = cross_sell_types)
        df_ts_no_targ = df_ts.copy()
        
        # create cross-effects dummies
        df_ts_no_targ['business_retail_joint'] = df_no_targ['business'].copy()* df_no_targ['retail'].copy()*df_no_targ['joint'].copy()
        df_ts_no_targ['business_retail'] = df_no_targ['business'].copy()* df_no_targ['retail'].copy()*(1 - df_no_targ['joint'].copy())
        df_ts_no_targ['business'] = df_no_targ['business'].copy()* (1 - df_no_targ['retail'].copy())*(1 - df_no_targ['joint'].copy())
        df_ts_no_targ['business_joint'] = df_no_targ['business'].copy()*(1 - df_no_targ['retail'].copy())*(df_no_targ['joint'].copy())
        df_ts_no_targ['joint'] = (1 - df_no_targ['business'].copy())* (1 - df_no_targ['retail'].copy())*(df_no_targ['joint'].copy())  
        df_ts_no_targ['retail_joint'] = (1-df_no_targ['business'].copy())* (df_no_targ['retail'].copy())*(df_no_targ['joint'].copy())
        df_ts_no_targ['retail'] = (1-df_no_targ['business'].copy())* (df_no_targ['retail'].copy())*(1 -df_no_targ['joint'].copy())
        df_ts_no_targ['constant'] = np.ones(len(df_ts_no_targ))  #[1]* df_ts.shape[0]
    
        #add a few variables which are not standard in the dataset
        df_ts_no_targ['log_saldoprev'] = df_ts_no_targ["log_saldototaal"].copy()
        df_ts_no_targ['prev_retail_dummy'] = df_ts_no_targ["retail_dummy"].copy()
        df_ts_no_targ['prev_joint_dummy'] = df_ts_no_targ["joint_dummy"].copy()
        df_ts_no_targ['prev_business_dummy'] = df_ts_no_targ["business_dummy"].copy()
        
        
        # train model to get parameters
        if (isinstance(X_var_final, type(None))) or (isinstance(ols_final, type(None))):
            X_var_final, ols_final, r2adjusted, r2, mse = self.train_predict(test_set_prop = test_set_prop,
                                                                             random_state = random_state,
                                                                             p_bound = p_bound)    
            
            # use significant variables and corresponding parameters
            X_var_final = pd.Series(X_var_final)
            
            self.df_ts_final_cs = df_ts_cs[X_var_final]
            self.df_ts_final_no_targ = df_ts_no_targ[X_var_final]
            
            # calculate fitted values
            fitted_values_cs = ols_final.predict(self.df_ts_final_cs)
            fitted_values_no_targ = ols_final.predict(self.df_ts_final_no_targ)

            return fitted_values_cs, fitted_values_no_targ, X_var_final, ols_final
        else:
            # use significant variables and corresponding parameters
            X_var_final = pd.Series(X_var_final)
            
            df_ts_final_cs = df_ts_cs[X_var_final]
            df_ts_final_no_targ = df_ts_no_targ[X_var_final]
            
            # calculate fitted values
            fitted_values_cs = ols_final.predict(df_ts_final_cs)
            fitted_values_no_targ = ols_final.predict(df_ts_final_no_targ)

            return fitted_values_cs, fitted_values_no_targ


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
    

        prev_saldo = df["saldototaal"]
        #self.df[[["business","retail","joint"]].sum(axis=1)]
        
        new_saldo = np.exp(fitted_values) * (prev_saldo - minimum) + minimum
        
        extra_saldo = new_saldo - prev_saldo
        
        return extra_saldo
    
    
    def get_extra_saldo(self, cross_sell_total, cross_sell_self, time, minimum, fin_segment = None, 
                        test_set_prop = 0.2, random_state = 0, p_bound = 0.05, 
                        X_var_final = None, ols_final = None):
        """
        Parameters
        ----------
        cross_sell_yes_no : 2D array
            array indicating whether customers (rows) are eligible for the cross sell of certain products (columns)
        time : int
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

        if (isinstance(fin_segment, type(None))):
            df_ts = self.df_time_series[time-1]
        else:
            df_ts = self.df_time_series[time-1]     
            df_ts = df_ts[df_ts['finergy_tp'] == fin_segment]
            
        summation = np.sum(cross_sell_total, axis = 1)
        indices_cross_sell = np.nonzero(summation)
        
        df_ts_subset = df_ts.loc[indices_cross_sell].copy()
        cross_sell_self = cross_sell_self[indices_cross_sell]
        cross_sell_total = cross_sell_total[indices_cross_sell]

        
        if (isinstance(X_var_final, type(None))) or (isinstance(ols_final, type(None))):
            fitted_values_cs, fitted_values_no_targ,  X_var_final, ols_final =  self.get_fitted_values(cross_sell_types = self.cross_sell_types ,
                                                                                                       cross_sell_total = cross_sell_total,
                                                                                                       cross_sell_self = cross_sell_self,
                                                                                                       df_ts = df_ts,
                                                                                                       test_set_prop = test_set_prop,
                                                                                                       random_state = random_state,
                                                                                                       p_bound= p_bound)
            
            extra_saldo_cs = self.fitted_values_to_saldo(minimum, fitted_values_cs, df = df_ts)
            extra_saldo_no_targ = self.fitted_values_to_saldo(minimum, fitted_values_no_targ, df = df_ts)
            
            return extra_saldo_cs, extra_saldo_no_targ, indices_cross_sell, X_var_final, ols_final

        else: 
            fitted_values_cs, fitted_values_no_targ =  self.get_fitted_values(cross_sell_types = self.cross_sell_types ,
                                                                                cross_sell_total = cross_sell_total,
                                                                                cross_sell_self = cross_sell_self,                                                                            df_ts = df_ts,
                                                                                test_set_prop = test_set_prop,
                                                                                random_state = random_state,
                                                                                p_bound= p_bound,
                                                                                X_var_final = X_var_final,
                                                                                ols_final = ols_final)
    
            extra_saldo_cs = self.fitted_values_to_saldo(minimum, fitted_values_cs, df = df_ts_subset)
            extra_saldo_no_targ = self.fitted_values_to_saldo(minimum, fitted_values_no_targ, df = df_ts_subset)
            
            return extra_saldo_cs, extra_saldo_no_targ, indices_cross_sell
    


