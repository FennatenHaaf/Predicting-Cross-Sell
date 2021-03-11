"""
Class to transform data and perform analysis on a cross-section of Knab Data.
"""
from imp import new_module

from pipenv.vendor.contextlib2 import redirect_stderr

import utils
import declarationsFile
import dataInsight
import additionalDataProcess

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

import re
import importlib
import gc

class MachineLearningModel(object):

    def __init__(self, indir, interdir,outdir, plot_results = True,automatic_folder_change = False, do_print_logs = True):
        """
        Initialize the method to create a Cross Section Model and To model.
        """

        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir
        self.automatic_folder_change = automatic_folder_change
        self.seed = 978391 #Fixed seed
        self.plot_results = plot_results
        self.transformation_log = []
        self.do_print_logs = do_print_logs

        ###Current default Base year###
        self.first_date = None

        ###Values to set at begin
        self.set_kfold()

        #For debugging purposes
        self.force_error = False
        #Empty lists to parse if non existant
        self.imported_data_name = None
        self.loaded_split = None
        self.loaded_variable_splits = None

    def set_kfold( self , n_splits = 5):
        self.kfold = StratifiedKFold(random_state = self.seed, shuffle = True, n_splits = n_splits)

    def load_split(self, split_to_load):
        assert not self.force_error, "Forced Error"
        """
        Currently three datasets and estimations:
        Retail -> Has Joint or Not
        Retail (+values of Joint?) -> Has Business or not
        Retail + Business -> Has overlay or not (as every overlay has business connected)
        """
        splitnames = ['retail_only', 'joint_and_retail', 'joint_or_retail','retail_and_business']
        assert split_to_load in splitnames, f"choose a value from {splitnames}"

        if self.imported_data_name != 'cross_long_df':
            self.import_data_to_model(import_command = "cross_long_df")

        self.imported_data.sort_index(axis = 1, inplace = True)

        #TODO create loop to get correct column values

        general_exclusion_list = ['delta,aantalproducten,increased','prtf', 'dummy','change','benchmark']

        business_column_to_use = ['business','sbicode','sbiname','sbisector','aantal_types', 'saldototaal_fr','aantal_sbi',
                                  'aantal_sector']
        business_exclusion_list = [] + general_exclusion_list

        joint_column_to_use = ['joint']
        joint_exclusion_list = [] + general_exclusion_list

        retail_column_to_use = ['retail','educat4','age','child','hh_size','housetype','huidigewaarde','income']
        retail_exclusion_list =  ['businessage'] + general_exclusion_list

        #Find correct columns to use
        business_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns,business_column_to_use,
                                                                     business_exclusion_list)
        joint_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns, joint_column_to_use,
                                                                   joint_exclusion_list)
        retail_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns, retail_column_to_use,
                                                                   retail_exclusion_list)

        drop_list_general = ['period','birthyear']
        drop_exclusion = []
        drop_list_general = utils.do_find_and_select_from_list(self.imported_data,drop_list_general,drop_exclusion)
        ##Replace th

        if split_to_load == 'retail_only':
            columns_to_drop = business_column_to_use + joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query("has_ret_prtf == 1")
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        elif split_to_load in ['joint_and_retail','joint_or_retail']:
            columns_to_drop = business_column_to_use + drop_list_general
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)
            if split_to_load == 'joint_and_retail':
                self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_jnt_prtf == 1")
            else:
                self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 | has_jnt_prtf == 1")
        elif split_to_load == 'retail_and_business':
            columns_to_drop = joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_bus_prtf == 1", inplace = True)
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        self.transformation_log.append(f"{utils.print_seperated_list(columns_to_drop)} dropped for {split_to_load}")
        print(self.transformation_log[-1])

        self.loaded_split = split_to_load
        self.variables_dict = {'retail_column_to_use':retail_column_to_use,'joint_column_to_use':
            joint_column_to_use,
                               'business_column_to_use': business_column_to_use}

        print(f"Load split for '{split_to_load}' has been loaded at {utils.get_time()}")

    def join_together_retail_joint( self ):
        assert not self.force_error, "Forced Error"

        if self.loaded_split != 'joint_or_retail':
            self.load_split('joint_or_retail')
        retail_vars = self.variables_dict['retail_column_to_use']
        joint_vars = self.variables_dict['joint_column_to_use']
        new_retail_vars = []
        new_joint_vars = []
        for item in retail_vars:
            new_item = item.replace("_retail","")
            new_retail_vars.append(new_item)

        for item in joint_vars:
            new_item = item.replace("_joint","")
            new_joint_vars.append(new_item)

        intersected = set(new_retail_vars) & set(new_joint_vars)

        unique_vals = self.imported_data.nunique()
        binary_vals_list = ['sparenyn','depositoyn','betalenyn']
        intersected_no_binary = utils.do_find_and_select_from_list(intersected,[""],binary_vals_list)

        new_retail_vars = []
        new_joint_vars = []
        joint_retail_vars = []
        for item in intersected_no_binary:
            joint_retail_vars.append(f"{item}_retail_joint")
            new_retail_vars.append(f"{item}_retail")
            new_joint_vars.append((f"{item}_joint"))
            self.imported_data[[ new_retail_vars[-1],new_joint_vars[-1] ]] = self.imported_data[[
                new_retail_vars[-1],new_joint_vars[-1] ]].fillna(0,axis = 1)
            self.imported_data[joint_retail_vars[-1]] = self.imported_data[new_retail_vars[-1]]\
                                                           + self.imported_data[new_joint_vars[-1]]

        intersected_binary = list( set(intersected) - set(intersected_no_binary) )
        for item in intersected_binary:
            joint_retail_vars.append(f"{item}_retail_joint")
            new_retail_vars.append(f"{item}_retail")
            new_joint_vars.append((f"{item}_joint"))
            self.imported_data[[ new_retail_vars[-1],new_joint_vars[-1] ]]= self.imported_data[[ new_retail_vars[-1],
                                                                                                 new_joint_vars[-1] ]].fillna(0,axis = 1)
            self.imported_data[joint_retail_vars[-1]] = np.where( (self.imported_data[new_retail_vars[-1]]\
                                                           + self.imported_data[new_joint_vars[-1]]) > 0, 1, 0)

        exclude_drop_columns = ['aantalproducten','delta',]
        drop_columns = utils.do_find_and_select_from_list((new_joint_vars + new_retail_vars), [""],exclude_drop_columns)
        self.imported_data = self.imported_data.drop(drop_columns, axis = 1)
        pass

    def split_variable_sets( self, test_string ):
        splitnames = ['retail_only', 'joint_and_retail', 'joint_or_retail','retail_and_business']
        assert test_string in splitnames, f"choose a value from {splitnames}"
        assert not self.force_error, "Forced Error"

        if self.loaded_split != test_string:
            self.load_split(test_string)

        if self.loaded_split == 'joint_or_retail':
            self.join_together_retail_joint()

        old_columns = self.imported_data.columns
        self.imported_data = self.imported_data.dropna(axis = 1)
        self.transformation_log.append(f"Variables dropped for na values are "
                                       f" { utils.print_seperated_list( list(set(old_columns) - set(self.imported_data.columns)) ) } ")
        print(self.transformation_log[-1])

        id_var = ['personid']
        dependent_search_list = ['has_','increased','prtf', 'benchmark', 'delta','aantalproducten']
        exclude_general_list = []
        binary_exclusion_list = []
        exclude_categorical_list = ['total_counts']

        print(f"Selecting variables for {test_string}")
        if test_string == 'retail_only':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list

        elif test_string in ['joint_and_retail','joint_or_retail']:
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list

        elif test_string == 'retail_and_business':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list

        ##Selection of variables and
        dependent_variables = utils.do_find_and_select_from_list(self.imported_data.columns, dependent_search_list,
                                                                 dependent_exclusion_list)
        self.transformation_log.append(f"The Following Variables are possible dependent variables (length"
                                 f" {len(dependent_variables)}): \n {dependent_variables}\n")
        print(self.transformation_log[-1])

        unique_vals = self.imported_data.nunique()

        binary_variables = list(set(unique_vals[unique_vals <= 2].index) - set(dependent_variables) - set(binary_exclusion_list))
        self.transformation_log.append(f"The Following Variables have one value or are binary (length {len(binary_variables)}): \n"
            f" {binary_variables}\n")
        print(self.transformation_log[-1])


        categorical_variables = list(set(unique_vals[unique_vals <= 35].index))
        categorical_variables = list(set(categorical_variables) - set(dependent_variables) - set(binary_variables) - set(exclude_categorical_list))
        self.transformation_log.append(f"The Following Variables are categorical (length"
                                 f" {len(categorical_variables)}): \n {categorical_variables}\n")
        print(self.transformation_log[-1])


        value_variables = list( set(self.imported_data.columns) - set(dependent_variables + binary_variables + categorical_variables +
                                                                  id_var) )
        self.transformation_log.append(f"The Following Variables are either float or integer  (lengt" \
                                    f"h {len(value_variables)}): \n {value_variables}\n")
        print(self.transformation_log[-1])

        total_variables = dependent_variables  + binary_variables + categorical_variables + value_variables
        templist = list(set(self.imported_data.columns) - set(total_variables) - set(id_var))
        self.transformation_log.append(f"Not categorized variables are (length {len(templist)}): \n {templist}\n")
        print(self.transformation_log[-1])

        self.loaded_variable_splits = test_string
        self.variables_dict = {**self.variables_dict, 'dependent_variables':dependent_variables, 'binary_variables':binary_variables,
                            'categorical_variables':categorical_variables, 'value_variables':value_variables}
        self.imported_data.sort_index(axis = 1, inplace = True)
        pass


    def test_and_transform_static_variables( self , test_string, dependent_variable):
        splitnames = ['retail_only', 'joint_and_retail','joint_or_retail', 'retail_and_business']
        assert test_string in splitnames, f"choose a value from {splitnames}"
        assert not self.force_error, "Forced Error"

        if self.loaded_variable_splits != test_string:
            self.split_variable_sets(test_string)

        self.variables_dict['value_variables_with_classes'] = []
        self.variables_dict['value_variables_coded'] = []

        current_dependent_variable = dependent_variable
        for variable in self.variables_dict['value_variables']:
            self.transform_value_variables(self.imported_data, variable, current_dependent_variable)

        unordered_search_string = ['finergy','geslacht']
        unordered_variables = utils.do_find_and_select_from_list(self.imported_data.columns,unordered_search_string)

        self.variables_dict["categorical_variables_recoded"] = []
        self.variables_dict["categorical_variables_with_dummies"] = []
        is_unordered = False
        for variable in self.variables_dict['categorical_variables']:
            if variable in unordered_variables:
                is_unordered = True
            self.transform_categorical_variables(self.imported_data, variable, current_dependent_variable, 200,
                                                 is_unordered = is_unordered)
            is_unordered = False

        pass

    def transform_value_variables( self, data_to_use, x_variable, y_variable ):
        assert not self.force_error, "Forced Error"

        #Can add n_jobs = -1 to improve speed
        X = data_to_use.loc[:, x_variable].to_frame()
        y = data_to_use.loc[:, y_variable].to_frame()

        parameter_grid = {'max_depth': [1,2,3,4,5], 'min_samples_leaf': [20]}
        optimize_split = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid =parameter_grid, cv = self.kfold)
        optimize_split.fit(X,y)
        best_fit = optimize_split.best_estimator_.fit(X,y)


        if self.plot_results:
            plot_tree(best_fit, feature_names = [x_variable], class_names =  [y_variable], node_ids = True)
            plt.title = f"Tree for {x_variable} with dependent var {y_variable}"
            plt.show()

        if best_fit.tree_.n_leaves > 1:
            x_variable_class = f"{x_variable}_class"

            thresholds = []
            for threshold_value in best_fit.tree_.threshold:
                if threshold_value > 0:
                    thresholds.append(threshold_value)

            self.variables_dict['value_variables_coded'].append(x_variable_class)
            self.transformation_log.append(f"Added classificated variable : {x_variable_class} \n with"
                                           f"threshold values \n {utils.print_seperated_list(thresholds)}")
            self.print_transformation_log()

            X['classified_to'] = best_fit.apply(X)
            max_for_class = X.groupby(by = 'classified_to', as_index = False)[x_variable].max()
            max_for_class.sort_values(x_variable, inplace = True)

            self.imported_data[x_variable_class] = np.nan
            for i,item in enumerate(max_for_class.classified_to):
                self.imported_data.loc[X['classified_to'] == item, x_variable_class ] = int(i)

            if max_for_class.classified_to.shape[0] > 2:
                new_data = pd.get_dummies(self.imported_data[x_variable_class], prefix = x_variable_class, prefix_sep = "_")
                self.variables_dict['value_variables_with_classes'] + list(new_data.columns)
                self.transformation_log.append(f"Added dummies :{utils.print_seperated_list(new_data.columns)}")
                self.print_transformation_log()
                self.imported_data = pd.concat([self.imported_data,new_data], axis = 1)

        else:
            dropmessage = f"{x_variable}_class will not be created"
            self.transformation_log.append(dropmessage)
            self.print_transformation_log()

    def transform_categorical_variables(self , data_to_use, x_variable, y_variable, threshold, is_unordered):
        assert not self.force_error, "Forced Error"
        X = data_to_use.loc[:, x_variable].to_frame()
        y = data_to_use.loc[:, y_variable].to_frame()
        transformation_log = f"Transforming categorical {x_variable}"


        frequencies = X[x_variable].value_counts()
        high_frequency = frequencies > threshold

        other_category = frequencies[~high_frequency].index.tolist()
        high_frequency_category = frequencies[high_frequency].index.tolist()
        number_of_values = len(high_frequency_category)

        if is_unordered:

            new_data = pd.DataFrame(self.imported_data[x_variable])
            if len(other_category) > 1:
                new_data.loc[X[x_variable].isin(other_category), x_variable] = "Other"
                self.imported_data[f"{x_variable}_recoded"] = new_data[x_variable]
                self.variables_dict["categorical_variables_recoded"] = self.variables_dict["categorical_variables_recoded"] +\
                                                                       [f"{x_variable}_recoded"]

                transformation_log = f"{transformation_log}\n Categories coded with Other are: " \
                                     f" {utils.print_seperated_list(other_category)}"
            else:
                new_data.loc[X[x_variable] == high_frequency_category[-1], x_variable] = "Other"
                transformation_log = f"{transformation_log}\n Categories coded with Other are: " \
                                     f" |{high_frequency_category[-1]}|"

            new_data = pd.get_dummies( new_data, prefix = f"{x_variable}" )
            new_data = new_data.drop(f"{x_variable}_Other", axis = 1)
            self.variables_dict['categorical_variables_with_dummies'] = self.variables_dict['categorical_variables_with_dummies'] + \
                                                                        list(new_data.columns)
            self.imported_data = pd.concat([self.imported_data,new_data], axis = 1)

        if not is_unordered:
            max_dum = 6
            if (len(other_category) > 0) or (max_dum < frequencies.shape[0]):
                x_variable_recoded = f"{x_variable}_recoded"

                new_freqs = pd.DataFrame({'freq'      : frequencies.values, 'low_bound': frequencies.index,
                                                'high_bound': frequencies.index})
                while (new_freqs['freq'].min() < threshold) or (max_dum < new_freqs.shape[0]):
                    available_freqs = new_freqs.copy()
                    possible_combinations = pd.DataFrame(columns = ['freq', 'low_bound', 'high_bound'])

                    while available_freqs.shape[0] > 1:
                        lowest_index = available_freqs['freq'].argmin()
                        low_bound = int(available_freqs.loc[lowest_index,'low_bound'])
                        high_bound = int(available_freqs.loc[lowest_index,'high_bound'])
                        new_low = int(low_bound - 1)
                        new_high = int(high_bound + 1)
                        min_value = available_freqs.loc[lowest_index,'freq']
                        available_freqs.drop(lowest_index, inplace = True)

                        if new_low in available_freqs['high_bound'].to_list():
                            new_index = available_freqs['high_bound'] == new_low
                            low_value = int(available_freqs.loc[new_index, 'freq'])
                            new_low = int(available_freqs.loc[new_index, 'low_bound'])
                        else:
                            low_value = np.inf
                        if new_high in available_freqs['low_bound'].to_list():

                            new_index = available_freqs['low_bound'] == new_high
                            high_value = int(available_freqs.loc[new_index, 'freq'])
                            new_high = int(available_freqs.loc[new_index, 'high_bound'])
                        else:
                            high_value = np.inf

                        if low_value < high_value:
                            new_combination = pd.DataFrame({'freq': (low_value + min_value) , 'low_bound': new_low ,
                                                             'high_bound': high_bound}, index = [0])
                        elif low_value >= high_value and (high_value < np.inf):
                            new_combination = pd.DataFrame({'freq': (high_value + min_value) , 'low_bound': low_bound ,
                                                             'high_bound': new_high}, index = [0])
                        else:
                            continue
                        possible_combinations = pd.concat([possible_combinations,new_combination], axis = 0, ignore_index = True)

                    min_index = possible_combinations['freq']).argmin()
                    low_bound = int(possible_combinations.loc[min_index,'low_bound'])
                    high_bound = int(possible_combinations.loc[min_index, 'high_bound'])

                    new_freqs = new_freqs[new_freqs['high_bound'] != low_bound]
                    new_freqs[new_freqs['low_bound'] == high_bound,['freq','low_bound','high_bound']] = possible_combinations.loc[
                        min_index,:]









                pass



        self.transformation_log.append(transformation_log)
        self.print_transformation_log()
        pass

    def run_random_forest_model( self ):
        RandomForestClassifier()


    ##IMPORTING DATA AND VARIABLE SETTERS
    def import_data_to_model(self, import_command, last_date = "", first_date = ""):
        process_data = additionalDataProcess.AdditionalDataProcess(self.indir,self.interdir,self.outdir)
        process_data.automatic_folder_change = self.automatic_folder_change
        if import_command == "cross_long_df":
            process_data.transform_to_different_sets("cross_long_df", first_date = first_date, last_date = last_date)
            self.imported_data = process_data.cross_long_df.copy()

        elif import_command == "panel_df":
            process_data.transform_to_different_sets("panel_df", first_date = first_date, last_date = last_date)
            self.imported_data = process_data.panel_df.copy()

        self.imported_data_name = import_command

    def set_dates(self, first_date, last_date):
        "Method for safely changing dates in this class"
        data_freq = utils.infer_date_frequency(first_date)
        assert data_freq != None \
               and (utils.infer_date_frequency(last_date) != None or\
               last_date == ""),\
            "No Valid date set"
        self.first_date = first_date
        self.current_freq = data_freq

        if last_date == "":
            self.last_date = first_date
            print(f"Period set to {self.first_date} with frequency {self.current_freq}")
        else:
            self.last_date = last_date
            print(f"Period set from {self.first_date} to {self.last_date} with frequency {self.current_freq}")

    def print_transformation_log( self, print_total = False ):
        if print_total:
            for i,log_item in enumerate(self.transformation_log):
                print(f"Log entry {i+1} :\n")
                print(f"{log_item}\n \n")
        else:
            if self.do_print_logs:
                print(self.transformation_log[-1])





    ###-----------------------MODELLING -----------------------------###
    def parameter_tuning(self):
        pass

    def compare_models(self):
        pass

    def train_and_predict_final_model(self):
        pass
