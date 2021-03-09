"""
Class to transform data and perform analysis on a cross-section of Knab Data.
"""
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

    def __init__(self, indir, interdir,outdir, plot_results = True,automatic_folder_change = False):
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
        ###Current default Base year###
        self.first_date = None

        ###Values to set at begin
        self.set_kfold()

        #Empty lists to parse if non existant
        self.imported_data_name = None
        self.total_variables = None

    def set_kfold( self , n_splits = 5):
        self.kfold = StratifiedKFold(random_state = self.seed, shuffle = True, n_splits = n_splits)

    def load_split(self, split_to_load):
        """
        Currently three datasets and estimations:
        Retail -> Has Joint or Not
        Retail (+values of Joint?) -> Has Business or not
        Retail + Business -> Has overlay or not (as every overlay has business connected)
        """
        splitnames = ['retail_only', 'joint_and_retail', 'retail_and_business']
        assert split_to_load in splitnames, f"choose a value from {splitnames}"

        if self.imported_data_name != 'cross_long_df':
            self.import_data_to_model(import_command = "cross_long_df")

        self.imported_data.sort_index(axis = 1, inplace = True)

        #TODO create loop to get correct column values
        business_column_to_use = ['business','sbicode','sbiname','sbisector','aantal_types', 'saldototaal_fr','aantal_sbi',
                                  'aantal_sector']
        business_exclusion_list = ['dummy','change','increase','prtf','increased']

        joint_column_to_use = ['joint']
        joint_exclusion_list = ['dummy','change','increase','prtf']

        retail_column_to_use = ['retail','educat4','age']
        retail_exclusion_list =  ['businessage','benchmark','increase','prtf']

        #Find correct columns to use
        business_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns,business_column_to_use,business_exclusion_list)
        joint_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns, joint_column_to_use,joint_exclusion_list)
        retail_column_to_use = utils.do_find_and_select_from_list(self.imported_data.columns, retail_column_to_use,retail_exclusion_list)

        drop_list_extra = ['period','birthyear']
        drop_exclusion = []
        drop_list_general = utils.do_find_and_select_from_list(self.imported_data,drop_list_extra,drop_exclusion)
        ##Replace th

        if split_to_load == 'retail_only':
            # TODO Zoek waar verkeerde waarden vandaan komen voor de laatste periode
            columns_to_drop = business_column_to_use + joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query("has_ret_prtf == 1").copy()
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        elif split_to_load == 'joint_and_retail':
            columns_to_drop = business_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_jnt_prtf == 1", inplace = True).copy()
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)
        elif split_to_load == 'retail_and_business':
            columns_to_drop = joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_bus_prtf == 1", inplace = True).copy()
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        self.imported_data.dropna(inplace = True)
        print(f"Load split for '{split_to_load}' has been loaded at {utils.get_time()}")


    def split_variable_sets( self, test_string ):
        splitnames = ['retail_only', 'joint_and_retail', 'retail_and_business']
        assert test_string in (splitnames + ['all']), f"choose a value from {splitnames  + ['all']}"

        if test_string == 'all':
            for loop_string in splitnames:
                self.split_variable_sets(test_string = loop_string)

        if self.imported_data_name != test_string:
            self.load_split(test_string)

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

        elif test_string == 'joint_and_retail':
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
        self.dependent_variables = utils.do_find_and_select_from_list(self.imported_data.columns, dependent_search_list,
                                                                 dependent_exclusion_list)
        print(f"The Following Variables are possible dependent variables (length {len(self.dependent_variables)}): \n {self.dependent_variables}\n")

        unique_vals = self.imported_data.nunique()
        unique_vals_sorted = unique_vals.sort_values()

        self.binary_variables = list(set(unique_vals[unique_vals <= 2].index) - set(self.dependent_variables) - set(binary_exclusion_list))
        print(f"The Following Variables have one value or are binary (length {len(self.binary_variables)}): \n {self.binary_variables}\n")

        self.categorical_variables = list(set(unique_vals[unique_vals <= 35].index))
        self.categorical_variables = list(set(self.categorical_variables) - set(self.dependent_variables) - set(self.binary_variables) - set(exclude_categorical_list))
        print(f"The Following Variables have one value or are categorical (length {len(self.categorical_variables)}): \n {self.categorical_variables}\n")

        self.value_variables = list( set(self.imported_data.columns) - set(self.dependent_variables + self.binary_variables + self.categorical_variables +
                                                                  id_var) )
        print(f"The Following Variables have one value or are categorical (length {len(self.value_variables)}): \n {self.value_variables}\n")

        self.total_variables = self.dependent_variables  + self.binary_variables + self.categorical_variables + self.value_variables
        templist = list(set(self.imported_data.columns) - set(self.total_variables) - set(id_var))
        message = f"Not categorized variables are (length {len(templist)}): \n {templist}\n"
        self.transformation_log.append(message)
        print(message)

        ##Create static features

    def test_and_transform_static_variables( self , test_string = ""):
        splitnames = ['retail_only', 'joint_and_retail', 'retail_and_business']
        assert test_string in (splitnames + ['all']), f"choose a value from {splitnames  + ['all']}"
        if self.total_variables == None:
            self.split_variable_sets(test_string)

        self.transformed_data = pd.DataFrame()
        current_dependent_variable = 'increased_business_prtf_counts'

        self.transformed_data[current_dependent_variable] = self.imported_data[current_dependent_variable]
        for variable in self.value_variables:
            self.transform_value_variables(self.imported_data, variable, current_dependent_variable)

        for variable in self.categorical_variables:
            self.transform_categorical_variables(self.imported_data, variable, current_dependent_variable, 200)
        pass



    def transform_value_variables( self, data_to_use, x_variable, y_variable ):
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
            X['classified_to'] = best_fit.apply(X)
            max_for_class = X.groupby(by = 'classified_to', as_index = False)[x_variable].max()
            max_for_class.sort_values(x_variable, inplace = True)

            self.transformed_data[x_variable] = np.nan
            for i,item in enumerate(max_for_class.classified_to):
                self.transformed_data.loc[X['classified_to'] == item, x_variable ] = i
        else:
            dropmessage = f"{x_variable} will be dropped"
            self.transformation_log.append(dropmessage)

        pass

    def transform_categorical_variables(self , data_to_use, x_variable, y_variable, threshold):
        X = data_to_use.loc[:, x_variable].to_frame()
        y = data_to_use.loc[:, y_variable].to_frame()
        transformation_log = f"Transforming categorical {x_variable}"

        frequencies = X[x_variable].value_counts()
        high_frequency = frequencies > threshold

        other_category = frequencies[~high_frequency].index.tolist()
        high_frequency_category = frequencies[high_frequency].index.tolist()

        number_of_values = len(high_frequency_category)
        self.transformed_data[x_variable] = np.nan


        for i,item in enumerate(high_frequency_category):
            self.transformed_data.loc[X[x_variable] == item, x_variable] == i
            transformation_log = f"{transformation_log} | {item} with value {i} "

        if len(other_category) >= 1:
            number_of_values = number_of_values - 1
            if len(other_category) == 1:
                transformation_log = f"{transformation_log} | {other_category[0]} with value {number_of_values} "
            else:
                transformation_log = f"{transformation_log}. Following values will be in category other :"\
                                     f"{utils.print_seperated_list(other_category)} "

            self.transformed_data.loc[X[x_variable].isin(other_category), x_variable] == (number_of_values - 1)
        self.transformation_log.append(transformation_log)
        print(self.transformation_log[-1])

    def run_random_forest_model( self ):
        RandomForestClassifier()










    ##IMPORTING DATA AND VARIABLE SETTERS
    def import_data_to_model(self, import_command, last_date = "", first_date = ""):
        process_data = additionalDataProcess.AdditionalDataProcess(self.indir,self.interdir,self.outdir)
        process_data.automatic_folder_change = self.automatic_folder_change
        if import_command == "cross_long_df":
            process_data.transform_to_different_sets("cross_long_df", first_date = first_date, last_date = last_date)
            self.imported_data = process_data.cross_long_df
            self.imported_data_name = import_command
        if import_command == "panel_df":
            process_data.transform_to_different_sets("panel_df", first_date = first_date, last_date = last_date)
            self.imported_data = process_data.panel_df
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



    ###-----------------------MODELLING -----------------------------###
    def parameter_tuning(self):
        pass

    def compare_models(self):
        pass

    def train_and_predict_final_model(self):
        pass
