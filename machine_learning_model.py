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

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


from datetime import datetime
from os import path
from os import mkdir
from shutil import copy2

import re
import importlib
import gc

class MachineLearningModel(object):

    def __init__( self, indir, interdir, outdir, plot_results = True, automatic_folder_change = False, do_print_logs = True,
                  archiving_active = True ):
        """
        Initialize the method to create a Cross Section Model and To model.
        """

        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir
        self.automatic_folder_change = automatic_folder_change
        self.seed = 978391 #Fixed seed
        self.seed2 = 417742
        self.plot_results = plot_results
        self.transformation_log = []
        self.do_print_logs = do_print_logs
        if archiving_active:
            self.create_archive_folders()

        ###Current default Base year###
        self.first_date = None

        #For debugging purposes
        #Empty lists to parse if non existant
        self.imported_data_name = None
        self.loaded_split = None
        self.loaded_variable_splits = None
        self.value_var_transformed = True
        self.value_cat_transformed = True

    def set_current_data_selection_to_use( self, data_selection_to_use):
        selectionnames = ['retail_only', 'joint_and_retail', 'joint_or_retail', 'retail_and_business']
        assert data_selection_to_use in selectionnames, f"choose a value from {selectionnames}"
        self.current_data_selection = data_selection_to_use

    def set_current_dependent_variable( self, dependent_variable_to_use ):
        if self.loaded_variable_splits != None:
            assert dependent_variable_to_use in self.imported_data.columns, "Dependent variable not found in columns"
        self.current_dependent_var = dependent_variable_to_use

    def run_scenario( self, part_to_test, run_variable_parsing, run_parameter_test, run_final_model, final_variables_input = None):

        final_variables_1 = ['hh_child',
                           'housetype',
                           'income',
                           'finergy_super_A',
                           'finergy_super_B',
                           'finergy_super_C',
                           'finergy_super_D',
                           'finergy_super_E',
                           'finergy_super_F',
                           'finergy_super_G',
                           'finergy_super_H',
                           'finergy_super_I',
                           'finergy_super_J',
                           'hh_size',
                           'activitystatus_retail_joint_recoded',
                           'finergy_tp_recoded_A01',
                           'finergy_tp_recoded_B04',
                           'finergy_tp_recoded_B05',
                           'finergy_tp_recoded_C06',
                           'finergy_tp_recoded_C07',
                           'finergy_tp_recoded_C09',
                           'finergy_tp_recoded_F16',
                           'finergy_tp_recoded_F17',
                           'finergy_tp_recoded_F18',
                           'finergy_tp_recoded_G21',
                           'finergy_tp_recoded_G22',
                           'finergy_tp_recoded_I25',
                           'finergy_tp_recoded_I26',
                           'finergy_tp_recoded_Other',
                           'huidigewaarde_klasse',
                           'geslacht_Man',
                           'geslacht_Man(nen) en vrouw(en)',
                           'geslacht_Vrouw',
                           'lfase',
                           'age_hh_recoded',
                           'activitystatus_retail_joint_recoded',
                           'educat4',
                           'logins_totaal_retail_joint_discrete',
                           'logins_totaal_retail_joint_discrete',
                           'aantalloginsapp_retail_joint_discrete',
                           'aantaltransacties_totaal_retail_joint_discrete',
                           'aantaltegenrekeningenlaatsteq_retail_joint_discrete',
                           'aantalbetaaltransacties_retail_joint_discrete',
                           'aantaltransacties_totaal_retail_joint_discrete',
                           'aantalfueltransacties_retail_joint_discrete',
                           'current_age_discrete',
                           'saldototaal_retail_joint_discrete',
                           'aantalloginsweb_retail_joint_discrete',
                           'saldototaal_discrete',
                           'aantalpostransacties_retail_joint_discrete',
                           'aantalatmtransacties_retail_joint_discrete']

        parameters_1 = {'criterion': 'gini','max_depth': 5,'max_features': 17,'min_samples_split': 20,'n_estimators': 100,
                        'random_state': 978391}

        if part_to_test == 'has_bus_prtf':
            self.set_current_data_selection_to_use("joint_or_retail")
            self.load_split(dataset_basis = 'cross_df')
            exclude_list_arg = ['finergy_tp','loginsapp','loginsweb']
            self.split_variable_sets(exclude_list_arg = exclude_list_arg)
            self.set_current_dependent_variable(part_to_test)
            self.transform_variables()
            if run_variable_parsing:
                self.test_general_importance_vars(n_to_use = 4)
            else:
                final_variables_1_clean = utils.doListIntersect(final_variables_1,self.imported_data.columns)
                self.variables_dict['final_variables'] = final_variables_1_clean
            if run_parameter_test:
                self.run_hyperparameter_tuning()
            else:
                self.parameters = parameters_1
            if run_final_model:
                self.run_final_model()


    def load_split(self, dataset_basis = 'cross_df'):
        """
        Currently three datasets and estimations:
        Retail -> Has Joint or Not
        Retail (+values of Joint?) -> Has Business or not
        Retail + Business -> Has overlay or not (as every overlay has business connected)
        """
        assert dataset_basis in ['cross_df','cross_long_df'], "Choose between 'cross_df' or 'cross_long_df' "

        if self.imported_data_name != dataset_basis:
            self.import_data_to_model(import_command = dataset_basis)

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

        if self.current_data_selection == 'retail_only':
            columns_to_drop = business_column_to_use + joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query("has_ret_prtf == 1")
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        elif self.current_data_selection in ['joint_and_retail','joint_or_retail']:
            columns_to_drop = business_column_to_use + drop_list_general
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)
            if self.current_data_selection == 'joint_and_retail':
                self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_jnt_prtf == 1")
            else:
                self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 | has_jnt_prtf == 1")
        elif self.current_data_selection == 'retail_and_business':
            columns_to_drop = joint_column_to_use + drop_list_general
            self.imported_data = self.imported_data.query(f"has_ret_prtf == 1 & has_bus_prtf == 1", inplace = True)
            self.imported_data.drop(columns_to_drop, axis = 1, inplace = True)

        self.transformation_log.append(f"{utils.print_seperated_list(columns_to_drop)} dropped for {self.current_data_selection}")
        print(self.transformation_log[-1])

        self.loaded_split = self.current_data_selection
        self.variables_dict = {'retail_column_to_use':retail_column_to_use,'joint_column_to_use':
            joint_column_to_use,
                               'business_column_to_use': business_column_to_use}

        print(f"Load split for '{self.current_data_selection}' has been loaded at {utils.get_time()}")



    def split_variable_sets( self, exclude_list_arg: list = []):

        if self.loaded_split != self.current_data_selection:
            self.load_split()

        if self.loaded_split == 'joint_or_retail':
            self.join_together_retail_joint()

        old_columns = self.imported_data.columns
        self.imported_data = self.imported_data.dropna(axis = 1)
        self.transformation_log.append(f"Variables dropped for na values are "
                                       f" { utils.print_seperated_list( list(set(old_columns) - set(self.imported_data.columns)) ) } ")
        print(self.transformation_log[-1])

        self.imported_data.set_index('personid', inplace = True)
        dependent_search_list = ['has_','increased','prtf', 'benchmark', 'delta','aantalproducten','total_counts']
        exclude_general_list = [] + exclude_list_arg
        binary_exclusion_list = [] + exclude_general_list
        exclude_categorical_list = ['total_counts'] + exclude_general_list
        numerical_exclusion_list = [] + exclude_general_list

        print(f"Selecting variables for {self.current_data_selection}")
        if self.current_data_selection == 'retail_only':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list
            exclude_categorical_list = [] + exclude_categorical_list
            numerical_exclusion_list = [] + numerical_exclusion_list

        elif self.current_data_selection in ['joint_and_retail','joint_or_retail']:
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list
            exclude_categorical_list = [] + exclude_categorical_list
            numerical_exclusion_list = [] + numerical_exclusion_list

        elif self.current_data_selection == 'retail_and_business':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list
            exclude_categorical_list = [] + exclude_categorical_list
            numerical_exclusion_list = [] + numerical_exclusion_list

        ##Selection of variables and
        dependent_variables = utils.do_find_and_select_from_list(self.imported_data.columns, dependent_search_list,
                                                                 dependent_exclusion_list)
        self.transformation_log.append(f"The Following Variables are possible dependent variables (length"
                                 f" {len(dependent_variables)}): \n {dependent_variables}\n")
        print(self.transformation_log[-1])

        unique_vals = self.imported_data.nunique()

        binary_variables = list(set(unique_vals[unique_vals <= 2].index) - set(dependent_variables) - set(binary_exclusion_list))
        binary_variables = utils.do_find_and_select_from_list(binary_variables,[""],binary_exclusion_list)
        self.transformation_log.append(f"The Following Variables have one value or are binary (length {len(binary_variables)}): \n"
            f" {binary_variables}\n")
        print(self.transformation_log[-1])


        categorical_variables = list(set(unique_vals[unique_vals <= 35].index))
        categorical_variables = list(set(categorical_variables) - set(dependent_variables) - set(binary_variables) - set(exclude_categorical_list))
        categorical_variables = utils.do_find_and_select_from_list(categorical_variables, [""], exclude_categorical_list)
        self.transformation_log.append(f"The Following Variables are categorical (length"
                                 f" {len(categorical_variables)}): \n {categorical_variables}\n")
        print(self.transformation_log[-1])


        numerical_variables = list( set(self.imported_data.columns) - set(dependent_variables + binary_variables + categorical_variables ) )
        self.transformation_log.append(f"The Following Variables are either float or integer  (lengt" \
                                    f"h {len(numerical_variables)}): \n {numerical_variables}\n")
        print(self.transformation_log[-1])

        total_variables = dependent_variables  + binary_variables + categorical_variables + numerical_variables
        templist = list(set(self.imported_data.columns) - set(total_variables))
        self.transformation_log.append(f"Not categorized variables are (length {len(templist)}): \n {templist}\n")
        print(self.transformation_log[-1])

        self.loaded_variable_splits = self.current_data_selection
        self.variables_dict = {**self.variables_dict, 'dependent_variables':dependent_variables, 'binary_variables':binary_variables,
                            'categorical_variables':categorical_variables, 'numerical_variables':numerical_variables}
        self.imported_data.sort_index(axis = 1, inplace = True)
        pass

    def join_together_retail_joint( self ):
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

    def transform_variables( self, type_variables_to_transform = "all" ):
        if self.loaded_variable_splits != self.current_data_selection:
            self.split_variable_sets()

        if type_variables_to_transform in ['numerical_variables','all']:
            self.variables_dict['numerical_variables_with_dummies'] = []
            self.variables_dict['numerical_variables_discrete'] = []


            for variable in self.variables_dict['numerical_variables']:
                self.transform_numerical_variables(variable)

            self.value_var_transformed = True

        if type_variables_to_transform in ['categorical_variables','all']:
            unordered_search_string = ['finergy','geslacht']
            unordered_variables = utils.do_find_and_select_from_list(self.imported_data.columns,unordered_search_string)

            self.variables_dict["categorical_variables_recoded"] = []
            self.variables_dict["categorical_variables_dummies"] = []
            self.variables_dict['categorical_variables_recoded_dummies'] = []


            for variable in self.variables_dict['categorical_variables']:
                if variable in unordered_variables:
                    is_unordered = True
                else:
                    is_unordered = False
                self.transform_categorical_variables(variable, 500, is_unordered = is_unordered)

            self.variables_dict['categorical_unordered'] = unordered_variables
            self.variables_dict['categorical_ordered'] = list(set(self.variables_dict['categorical_variables']) - set(
                unordered_variables))

            self.value_cat_transformed = True

    def transform_numerical_variables( self, x_variable, n_splits = 5 ):
        #Can add n_jobs = -1 to improve speed
        y_variable = self.current_dependent_var

        X = self.imported_data.loc[:, x_variable].to_frame()
        y = self.imported_data.loc[:, y_variable].to_frame()

        kfold = StratifiedKFold(random_state = self.seed, shuffle = True, n_splits = n_splits)
        parameter_grid = {'max_depth': [1,2,3], 'min_samples_leaf': [50] }
        optimize_split = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 465427), param_grid =parameter_grid,
                                                                         cv = kfold)
        optimize_split.fit(X,y)
        best_estimator = optimize_split.best_estimator_
        best_estimator.fit(X,y)

        "Draw and/or save plot of results"
        if self.plot_results or self.archiving_active:
            try:
                plot_tree(best_estimator, feature_names = x_variable, class_names =  y_variable, node_ids = True)
            except:
                pass
            plt.title = f"Tree for {x_variable} with dependent var {y_variable}"
            if self.archiving_active:
                self.save_plots()
            if self.plot_results:
                plt.show()

        if best_estimator.n_classes_ > 1:
            x_variable_discrete = f"{x_variable}_discrete"

            thresholds = []
            for threshold_value in best_estimator.tree_.threshold:
                if threshold_value > 0:
                    thresholds.append(threshold_value)
            thresholds.sort()

            self.variables_dict['numerical_variables_discrete'].append(x_variable_discrete)
            self.transformation_log.append(f"Added discretized variable : {x_variable_discrete} \n with"
                                           f"threshold values \n {utils.print_seperated_list(thresholds)}")
            self.print_transformation_log()

            """"Create new variable to contain discretized numerical value and make sure that a higher bin value is 
            associated with higher numerical values"""
            X['classified_to'] = best_estimator.apply(X)
            self.imported_data[x_variable_discrete] = X['classified_to']
            self.imported_data[x_variable_discrete] = self.imported_data[x_variable_discrete] - 1
            minmax_for_discrete = X.groupby(by = 'classified_to', as_index = False)[x_variable].aggregate( ['min','max'] )
            minmax_for_discrete.sort_values('min', inplace = True)
            minmax_for_discrete.reset_index(drop = True, inplace = True)

            self.imported_data[x_variable_discrete] = 0
            for i in range(0,minmax_for_discrete.shape[0]):
                change_var_index = (self.imported_data[x_variable] >= minmax_for_discrete.loc[i, 'min']) &\
                                   (self.imported_data[x_variable] <= minmax_for_discrete.loc[i,'max'])
                self.imported_data.loc[change_var_index, x_variable_discrete] = int(i)

            """"If more than two classes, additional dummy values are required for dummy specification"""
            if best_estimator.n_classes_ > 2:
                new_data = pd.get_dummies(self.imported_data[x_variable_discrete], prefix = x_variable_discrete)
                self.variables_dict['numerical_variables_with_dummies'] =self.variables_dict['numerical_variables_with_dummies'] \
                                                                         + list(new_data.columns)

                self.transformation_log.append(f"Added dummies :{utils.print_seperated_list(new_data.columns)}")
                self.print_transformation_log()
                self.imported_data = pd.concat([self.imported_data,new_data], axis = 1)

        else:
            dropmessage = f"{x_variable}_discrete will not be created"
            self.transformation_log.append(dropmessage)
            self.print_transformation_log()

    def transform_categorical_variables(self , x_variable, threshold, is_unordered, max_cats = 5):
        y_variable = self.current_dependent_var
        X = self.imported_data.loc[:, x_variable].to_frame()
        y = self.imported_data.loc[:, y_variable].to_frame()
        x_variable_recoded = f"{x_variable}_recoded"

        start_message = f"Starting Transforming categorical {x_variable}"
        self.transformation_log.append(start_message)
        self.print_transformation_log()
        has_been_recoded = False

        frequencies = X[x_variable].value_counts()
        high_frequency = frequencies > threshold

        other_category = frequencies[~high_frequency].index.tolist()
        high_frequency_category = frequencies[high_frequency].index.tolist()
        number_of_values = len(high_frequency_category)

        if is_unordered:
            """
            UNORDERED VARIABLES - Creates a new 
            """

            new_data = pd.DataFrame(self.imported_data[x_variable])
            if len(other_category) > 1:
                new_data.loc[X[x_variable].isin(other_category), x_variable] = "Other"
                self.imported_data[x_variable_recoded] = new_data[x_variable]
                self.variables_dict["categorical_variables_recoded"].append(x_variable_recoded)
                self.transformation_log.append(f"\n Categories coded with Other are: {utils.print_seperated_list(other_category)}\n")
                has_been_recoded = True

        else:
            """
            ORDERED OR ORDER LIKE VARIABLES. THIS PART CREATES LARGER CATEGORIES BY MERGING
            SMALLER CATEGORIES. A NEW GROUP IS CREATED BY ADDING AN ADJACENT CATEGORY. FOR EVERY
            ITERATION THE SMALLEST POSSIBLE NEW GROUP IS ADDED UNTIL THE VALUES ARE ABOVE
            THE VALUE THRESHOLD AND BELOW THE MAX NUMBER OF CATEGORIES
            """

            if (len(other_category) > 0) or (max_cats < frequencies.shape[0]):
                """
                FOR ORDERED VARIABLES WITH FREQUENCIES BELOW THRESHOLD OR ABOVE MAX CATEGORIES
                """
                x_variable_recoded = f"{x_variable}_recoded"

                new_freqs = pd.DataFrame({'freq'      : frequencies.values, 'low_bound': frequencies.index,
                                                'high_bound': frequencies.index})
                print("\n",new_freqs,"\n")
                while (new_freqs['freq'].min() < threshold) or (max_cats < new_freqs.shape[0]):
                    available_freqs = new_freqs.copy()
                    best_combination = (np.inf,0,0)

                    while available_freqs.shape[0] > 1:
                        current_lowest_val = best_combination[0]
                        available_freqs.reset_index(drop = True, inplace = True)

                        lowest_index = int(available_freqs['freq'].argmin())
                        low_bound = int(available_freqs.loc[lowest_index,'low_bound'])
                        high_bound = np.int(available_freqs.loc[lowest_index,'high_bound'])
                        new_low = np.int(low_bound - 1)
                        new_high = np.int(high_bound + 1)
                        min_value = int(available_freqs.loc[lowest_index,'freq'])
                        available_freqs.drop(lowest_index, inplace = True)

                        if min_value > current_lowest_val:
                            break

                        if new_low in available_freqs['high_bound'].to_list():
                            new_index = available_freqs['high_bound'] == new_low
                            low_value = int(available_freqs.loc[new_index, 'freq']) + min_value
                            new_low = int(available_freqs.loc[new_index, 'low_bound'])
                        else:
                            low_value = np.inf

                        if new_high in available_freqs['low_bound'].to_list():
                            new_index = available_freqs['low_bound'] == new_high
                            high_value = int(available_freqs.loc[new_index, 'freq'] + min_value)
                            new_high = int(available_freqs.loc[new_index, 'high_bound'])
                        else:
                            high_value = np.inf

                        if (high_value > current_lowest_val) and (low_value > current_lowest_val):
                            continue
                        if low_value < high_value:
                            best_combination = (low_value,new_low,high_bound)
                        elif (low_value >= high_value) and (high_value != np.inf):
                            best_combination = (high_value,low_bound,new_high)
                        else:
                            continue

                    if best_combination[0] == np.inf:
                        break
                    low_bound = best_combination[1]
                    high_bound = best_combination[2]

                    freqs_to_drop = new_freqs['high_bound'] == high_bound
                    freqs_to_drop = freqs_to_drop | (new_freqs['low_bound'] == low_bound)
                    new_freqs = new_freqs[~freqs_to_drop].copy()
                    new_freqs.reset_index(drop = True, inplace = True)

                    new_index = new_freqs.shape[0]
                    for i,item in enumerate(new_freqs.columns.tolist()):
                        new_freqs.loc[new_index,item] = best_combination[i]

                    new_freqs.reset_index(drop =  True, inplace =  True)

                print("\n",new_freqs,"\n")
                recoded_list = []
                new_freqs.sort_values('low_bound', inplace = True)


                for i in range(0,new_freqs.shape[0]):
                    low_bound = new_freqs.iloc[i,1]
                    high_bound = new_freqs.iloc[i,2]
                    recoded_list.append( (low_bound,high_bound,(i+1)) )

                self.imported_data[x_variable_recoded ] = 0
                self.variables_dict['categorical_variables_recoded'].append(x_variable_recoded)
                self.transformation_log.append( f"{x_variable} has been recoded into {len(recoded_list)} categories" )
                self.print_transformation_log()

                transformation_log_add = ""
                for new_var in recoded_list:
                    recode_index = (self.imported_data[x_variable] >= new_var[0]) & (self.imported_data[x_variable] <= new_var[1])
                    self.imported_data.loc[recode_index,x_variable_recoded] = new_var[2]
                    transformation_log_add = f"{transformation_log_add} || Lower bound:{new_var[0]} , Upper bound {new_var[1]}  has " \
                                             f"been coded with < {new_var[2]} > ||"

                self.transformation_log.append(transformation_log_add)
                self.print_transformation_log()
                has_been_recoded = True

        """" If a variable has been recoded """
        if has_been_recoded:
            new_data = pd.get_dummies(self.imported_data[x_variable_recoded], prefix = f"{x_variable_recoded}")
            self.imported_data = pd.concat([self.imported_data, new_data], axis = 1)
            self.variables_dict['categorical_variables_recoded_dummies'] = self.variables_dict[
                                                                                 'categorical_variables_recoded_dummies'] \
                                                                         + list(new_data.columns)

        """FINAL CONCATENATION OF FILES"""
        new_data = self.imported_data[x_variable].copy()
        new_data = pd.get_dummies(new_data, prefix = f"{x_variable}")
        self.imported_data = pd.concat([self.imported_data, new_data], axis = 1)
        self.variables_dict['categorical_variables_dummies'] = self.variables_dict['categorical_variables_dummies'] \
                                                                         + list(new_data.columns)
        self.transformation_log.append(f"Added {new_data.columns.shape[0]} dummy categories for each value of {x_variable}. "
                                       f"Category transformation finished\n")
        self.print_transformation_log()

        pass

    def test_general_importance_vars(self, n_to_use = 4):
        y = self.imported_data[self.current_dependent_var]

        new_seed =  174011
        new_seed2 = 871124

        #Create one kfold split to be used in every instance
        split_to_use = []
        for train, test in  StratifiedKFold(n_to_use, shuffle = True, random_state = new_seed).split(np.zeros(self.imported_data.shape[
                                                                                                            0]), y):
            split_to_use.append( (train, test) )

        """
        Get different values to test for categorical values
        """
        cat_variables_to_test_dict = {}
        for variable in self.variables_dict['categorical_variables']:

            if variable in self.variables_dict['categorical_ordered']:
                cat_variables_to_test_dict[variable] = {variable: [variable] }
                for recoded_var in self.variables_dict['categorical_variables_recoded']:
                    if variable in recoded_var:
                        cat_variables_to_test_dict[variable] = {**cat_variables_to_test_dict[variable], recoded_var: [recoded_var]}
                        break
            else:
                cat_variables_to_test_dict[variable] = {}

            cat_var_list = []
            for dummy in self.variables_dict['categorical_variables_dummies']:
                if variable in dummy:
                    cat_var_list.append(dummy)
            if len(cat_var_list) > 1:
                cat_variables_to_test_dict[variable]= {**cat_variables_to_test_dict[variable], f'{variable}_dummies': cat_var_list }

            cat_var_list = []
            for dummy in self.variables_dict['categorical_variables_recoded_dummies']:
                if variable in dummy:
                    cat_var_list.append(dummy)
            if len(cat_var_list) > 1:
                cat_variables_to_test_dict[variable] = {**cat_variables_to_test_dict[variable], f'{variable}_recoded_dummies':
                    cat_var_list }


        """
        Get the numerical values to test
        """
        num_variables_to_test_dict = {}
        for variable in self.variables_dict['numerical_variables']:
            num_variables_to_test_dict[variable] = {variable: [variable]}
            for recoded_var in self.variables_dict['numerical_variables_discrete']:
                if variable in recoded_var:
                    num_variables_to_test_dict[variable] = {recoded_var: [recoded_var]}

            num_list = []
            for dummy_var in self.variables_dict['numerical_variables_with_dummies']:
                if variable in dummy_var:
                    num_list.append(dummy_var)
            if len(num_list) > 1:
                num_variables_to_test_dict[variable] = {f'{variable}_dummies': num_list}



        variables_to_use = self.variables_dict['binary_variables'] + \
                            self.variables_dict['categorical_ordered'] + self.variables_dict['numerical_variables']

        variables_testing_dic = {**cat_variables_to_test_dict, **num_variables_to_test_dict}


        """"
        Actual test for which variable to use in this analysis
        """
        final_variables = []
        for variable_to_test in variables_testing_dic:
            alternatives_to_test = variables_testing_dic[variable_to_test]
            if len(alternatives_to_test) == 1:
                alt_list = list(alternatives_to_test.keys())
                final_variables = final_variables + alternatives_to_test[ alt_list[0] ]
                self.transformation_log.append(f"{variable_to_test} has no alternative specifications able to be used, "
                                               f"{alt_list[0]} will be used for the default value")
                self.print_transformation_log()
                continue

            new_variables_to_use1 = list( set(variables_to_use) - set(variable_to_test) )
            mean_score_list = []
            for var_alternative in alternatives_to_test:
                new_variables_to_use2 = new_variables_to_use1 + alternatives_to_test[var_alternative]
                mean_score = 0
                x_selected = self.imported_data[new_variables_to_use2]
                transformationtext = f"For {var_alternative} it is found that the scores are | "
                for train, test in split_to_use:
                    y_train, y_test = y[train], y[test]
                    x_train, x_test = x_selected.iloc[train,:],\
                                      x_selected.iloc[test,:]

                    rf1 = RandomForestClassifier(random_state = new_seed2)
                    rf1.fit(x_train, y_train)
                    score_value = rf1.score(x_test, y_test)
                    mean_score += score_value
                    transformationtext = transformationtext + f"{score_value} | "


                mean_score = mean_score/n_to_use
                mean_score_list.append((var_alternative,mean_score, alternatives_to_test[var_alternative]))
                transformationtext = transformationtext + f"with mean score {mean_score}"
                self.transformation_log.append(transformationtext)
                self.print_transformation_log()

            best_alternative = (None, -1, [])
            for scored_alternative in mean_score_list:
                if scored_alternative[1] > best_alternative[1]:
                    best_alternative = (scored_alternative[0],scored_alternative[1],scored_alternative[2])
            final_variables = final_variables + best_alternative[2]

        self.variables_dict['final_variables'] = final_variables
        self.transformation_log.append(f"\nThe following variables were found to be the best to use in the final analysis: "
                                       f"{final_variables}\n")
        self.print_transformation_log()
        self.save_transformation_log()
        pass

    def run_hyperparameter_tuning( self, var_set = 1, n_splits = 4):
        """
        Run the 5x2 Cross Validation. Inner loop for tuning of hyperparameters and outer loop for training full results.
        """
        if self.value_var_transformed != self.current_data_selection:
            self.transform_variables(type_variables_to_transform = 'numerical_variables')
        if self.value_cat_transformed != self.current_data_selection:
            self.transform_variables(type_variables_to_transform = 'categorical_variables')

        #Set seed
        seed_new = 978391
        seed_new2 = 387293

        #Load personid to index
        # X = self.imported_data.set_index('personid')

        cols = self.variables_dict['binary_variables'] + self.variables_dict['final_variables']
        X = self.imported_data[cols]
        y = self.imported_data[self.current_dependent_var]
        n_of_vars = X.shape[1]


        parameters_for_gridsearch = {'criterion': ['gini'], 'max_features':[x for x in range(1,21)],'max_depth' : [1,2,3,4,5],
                                     'min_samples_split' : np.linspace(20,200, num = 5, dtype = 'int')  , 'n_estimators' :
                                         ( [50] +list(np.linspace(100,400,4, dtype = 'int')) ), 'random_state': [seed_new] }
        kfold = StratifiedKFold(random_state = seed_new2, shuffle = True, n_splits = n_splits)
        self.param_tuning = GridSearchCV(estimator = RandomForestClassifier(), param_grid =  parameters_for_gridsearch, cv = kfold,
                                    n_jobs = -1)
        self.param_tuning.fit(X,y)
        self.parameters = self.param_tuning.best_params_
        self.transformation_log.append(self.parameters)
        self.print_transformation_log()

        results = pd.DataFrame.from_dict(self.param_tuning.cv_results_)
        results.sort_values('mean_test_score', ascending = False, inplace = True)
        self.save_to_excel(data = results, addition_to_name = "_parameter_tuning")
        pass

    def run_final_model( self ):
        seed_new = 978391

        cols = self.variables_dict['binary_variables'] + self.variables_dict['final_variables']
        X = self.imported_data[cols]
        y = self.imported_data[self.current_dependent_var]

        final_rf1 = RandomForestClassifier(**self.parameters)
        feat_importance = final_rf1.feature_importances_
        importance_results_preliminary = pd.DataFrame(columns = ['variable_name','feature_importance'])
        for i,var in enumerate(X.columns):
            importance_results_preliminary.loc[i, 'variable_name'] = var
            importance_results_preliminary.loc[i,'feature_importance'] = feat_importance[i]

        importance_results_preliminary.sort_values('feature_importance', ascending = False,inplace = True)
        importance_results_preliminary.reset_index(drop = True, )

        with pd.option_context('display.max_rows', 500):
            print(importance_results_preliminary)

        self.save_to_excel(importance_results_preliminary,'_importance_prelim')

        # final_vars_to_use = importance_results_preliminary.iloc[0:20,:]
        # new_variables_to_use = utils.doListIntersect(final_vars_to_use['variable_name'],cols)
        #
        # X2 = self.imported_data[new_variables_to_use]




    """
    IMPORTING DATA AND VARIABLE SETTERS
    """
    def import_data_to_model(self, import_command, last_date = "", first_date = ""):
        """
        Method For Safely importing datasets into machine learning model
        """
        process_data = additionalDataProcess.AdditionalDataProcess(self.indir,self.interdir,self.outdir)
        process_data.automatic_folder_change = self.automatic_folder_change
        if import_command == "cross_df":
            process_data.transform_to_different_sets("cross_df", first_date = first_date, last_date = last_date)
            self.imported_data = process_data.cross_df.copy()

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

    def create_archive_folders( self ):
        start_time_string = datetime.now().strftime("%m%d_%H%M")
        "%m%d_%H%M%S"
        archive_dir = f"{self.outdir}/ML_results"
        if not path.exists(archive_dir):
            mkdir(archive_dir)
        self.subarchive_dir = f"{archive_dir}/{start_time_string}_results"
        mkdir(self.subarchive_dir)
        self.plotsdir = f"{self.subarchive_dir}/plots"
        mkdir(self.plotsdir)
        self.archiving_active = True

    def save_plots( self ):
        if not self.archiving_active:
            self.create_archive_folders()
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.subarchive_dir}/plotsoutput.pdf")
        pdf.savefig(plt.gcf())
        pdf.close()

        plot_time_string = datetime.now().strftime("%m%d_%H%M%S")
        plotname = f"{self.plotsdir}/{plot_time_string}_plot.png"
        plt.savefig(plotname)

    def save_to_excel( self , data,addition_to_name = ""):
        if not self.archiving_active:
            self.create_archive_folders()

        time_string = datetime.now().strftime("%m%d_%H%M%S")
        log_filename = f"{self.subarchive_dir}/{time_string}_results{addition_to_name}.xlsx"
        data.to_excel(log_filename, index = False)


    def save_transformation_log( self ):
        if not self.archiving_active:
            self.create_archive_folders()

        time_string = datetime.now().strftime("%m%d_%H%M%S")
        log_filename = f"{self.subarchive_dir}/{time_string}_transformation_log.txt"

        f = open(log_filename, "x")
        for log_entry in self.transformation_log:
            f.write(log_entry)
        f.close()

    def force_error( self ):
        for item in dir():
            if (item[0:2] != "__") and (item[0] != "_"):
                del globals()[item]
        # del self
