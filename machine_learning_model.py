"""
Class to transform data and perform analysis on a cross-section of Knab Data.

This code has not been used in the final analysis, but is a layout for possible further use
of a machine learning model.

The idea is that a large number of variables can be taken into account.
A general transformation based on the type of data wanted (Cross, Panel, Cross with delta based on dependent var etc)
will be done.

After that the variables in these datasets will be split into categories for further, category specific transformation
with several specifications for the same variable

After these transformations, several different variables

As input uses the data transformed by knab_dataprocessor after saved on the drive
"""
import utils

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from datetime import datetime
from datetime import date
from os import path
from os import mkdir

class MachineLearningModel(object):

    def __init__( self, indir, interdir, outdir, plot_results = True, automatic_folder_change = False, do_print_logs = True,
                  archiving_active = True ):
        """
        Initialize the method to create a Cross Section Model and To model.
        :param indir:
        :param interdir:
        :param outdir:
        :param plot_results:
        :param automatic_folder_change:
        :param do_print_logs:
        :param archiving_active:
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
        self.input_data = None
        self.transformed_df = None
        self.imported_data_name = None
        self.loaded_split = None
        self.loaded_variable_splits = None
        self.value_var_transformed = True
        self.value_cat_transformed = True


    def run_scenario( self, part_to_test, run_variable_parsing = False, run_parameter_test = False, run_final_model = False,
                      predefined_var:int= 0,  predefined_par:int = 0):
        """
        Run of the scenario's of Machine Learning Analysis
        :param part_to_test: Specific scenario to run
        :param run_variable_parsing: Run the part where different specification are tested
        tested for a certain measurement. If False, give a value for predefenid_var to run next methods/
        :param run_parameter_test: Run the tests for the parameters that will be used in the
        final model. If false -> give a value for predefined_par to parse final model.
        :param run_final_model: Run the final model.
        :param predefined_var: int for which predefined variable set to use from get_predefined_var()
        :param predefined_par: int for which predefined paramaters set to use from get_predefined_var()
        :return:
        """
        if not run_variable_parsing:
            final_variables = self.get_predefined_var(predefined_var)

        if not run_parameter_test:
            parameters = self.get_predefined_param(predefined_par)

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
                final_variables_1_clean = utils.doListIntersect(final_variables, self.transformed_df.columns)
                self.variables_dict['final_variables'] = final_variables_1_clean
            if run_parameter_test:
                self.run_hyperparameter_tuning()
            else:
                self.parameters = parameters
            if run_final_model:
                self.run_final_model()

    def transform_to_different_sets( self, transform_command, first_date = "", last_date = "" ):
        """
        Transform a set to a specific type of dataset with options:
        ['cross_df','cross_long_df', 'panel_df']

        self.transformed_data will be created
        :param transform_command: type of set to get
        :param first_date: first date for this set, if "" will use standard values
        :param last_date: last_date for this set, if "" will use standard value or
        first_date only (if a alternative first date is given as argument)
        :return:
        """
        if first_date == "":
            use_standard_period = True
        else:
            use_standard_period = False

        if transform_command in ['cross_df']:
            if use_standard_period:
                first_date, last_date = "2019Q1", "2019Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_cross_data()

        elif transform_command in ['cross_long_df']:
            if use_standard_period:
                first_date, last_date = "2018Q1", "2020Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_cross_data(last_date)
            self.add_longterm_change(benchmark_period = first_date)

        elif transform_command in ['panel_df']:
            if use_standard_period:
                first_date, last_date = "2018Q1", "2020Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_panel()

            self.input_data = None
            self.imported_data_name = transform_command

    def prepare_before_transform( self, first_date = "", last_date = "" ):
        """
        Prepare before transforming to the specific types of datasets
        :param first_date:
        :param last_date:
        :return:
        """
        print(f"Starting preparing data at {utils.get_time()}")
        if first_date == "":
            self.set_dates("2018Q1", "2020Q4")
        else:
            self.set_dates(first_date, last_date)

        # Import the different files related to the data to be prepared for transformation
        self.import_data('input_cross', first_date = self.first_date, last_date = self.last_date)

        #Create totals for the different business segments that can be found in the data
        for name in (["business","retail","joint"]):

            self.input_data[f'aantalproducten_totaal_{name}'] = self.input_data[[f"betalenyn_{name}",
                                        f"depositoyn_{name}",
                                        f"flexibelsparenyn_{name}",
                                        f"kwartaalsparenyn_{name}"]].sum(axis=1)

            self.input_data[f'logins_totaal_{name}'] = self.input_data[[f"aantalloginsapp_{name}",
                                        f"aantalloginsweb_{name}"]].sum(axis=1)

            self.input_data[f'aantaltransacties_totaal_{name}'] = self.input_data[[f"aantalbetaaltransacties_{name}",
                                        f"aantalatmtransacties_{name}",
                                        f"aantalpostransacties_{name}",
                                        f"aantalfueltransacties_{name}"]].sum(axis=1)


        # Take the MAX of the logins, transactions and activity status
        for var in ['logins_totaal','aantaltransacties_totaal','activitystatus']:
            self.input_data[f'{var}'] = self.input_data[[f"{var}_business", f"{var}_retail",
                                    f"{var}_joint"]].max(axis=1)

        # Sum for the total account balance
        self.input_data['saldototaal'] = self.input_data[["saldototaal_business", "saldototaal_retail",
                                "saldototaal_joint"]].sum(axis=1)

        # Also get total number of products
        self.input_data['aantalproducten_totaal'] = self.input_data[["aantalproducten_totaal_business",
                                           "aantalproducten_totaal_retail",
                                           "aantalproducten_totaal_joint"]].sum(axis=1)

        # Make lowercase names
        self.input_data.rename(str.lower, axis = 'columns', inplace = True)

        # Fill NA values with zeros to correctly set observations with not one of these options
        fillna_columns = ['business', 'joint', 'retail', 'accountoverlay', 'accountoverlay_dummy', 'accountoverlay_max']
        fillna_columns = utils.doListIntersect(fillna_columns, self.input_data.columns)
        self.input_data[fillna_columns] = self.input_data[fillna_columns].fillna(value = 0)

        ##RENAMING SEVERAL VARIABLES
        rename_dict = {
            'business'      : 'business_prtf_counts',
            'joint'         : 'joint_prtf_counts',
            'retail'        : 'retail_prtf_counts',
            'accountoverlay': 'aantalproducten_accountoverlay',
        }
        rename_dict = utils.doDictIntersect(self.input_data.columns, rename_dict)
        self.input_data.rename(rename_dict, axis = 1, inplace = True)

        today = date.today() #To use as imput for an age variable
        self.input_data = self.input_data.assign(
            #Create quarters and a year variable
            period_q2 = lambda x: np.where(x.period_obs.dt.quarter == 2, 1, 0),
            period_q3 = lambda x: np.where(x.period_obs.dt.quarter == 3, 1, 0),
            period_q4 = lambda x: np.where(x.period_obs.dt.quarter == 4, 1, 0),
            period_year = lambda x: x.period_obs.dt.year,
            #Create several variables as indicator if some has a retail, joint or
            #business portfolio or account overlay in a certain period
            has_ret_prtf = lambda x: np.where(x.retail_prtf_counts > 0, 1, 0),
            has_jnt_prtf = lambda x: np.where(x.joint_prtf_counts > 0, 1, 0),
            has_bus_prtf = lambda x: np.where(x.business_prtf_counts > 0, 1, 0),
            has_accountoverlay = lambda x: np.where(x.aantalproducten_accountoverlay > 0, 1, 0),
            #Total number of portfolios held
            portfolio_total_counts = lambda x: x.business_prtf_counts + x.joint_prtf_counts + x.retail_prtf_counts,
            #Current age of the person, age is in this case variable which takes less space than birthyear
            current_age = lambda x: today.year - x.birthyear,
            #The first letter of a finergy category, posssible replacement for finergy
            finergy_super = lambda x: x.finergy_tp.str[0]
        )

        if 'hh_size' in self.input_data.columns: #Change hhsize variable 10,11 to 1
            self.input_data.loc[self.input_data['hh_size'] == 11, 'hh_size'] = 1
            self.input_data.loc[self.input_data['hh_size'] == 10, 'hh_size'] = 1

        ##Change mannen to man and vrouwen to vrouw
        geslacht_var = utils.do_find_and_select_from_list(self.input_data.columns, ['geslacht'], [])
        for var in geslacht_var:
            self.input_data.loc[self.input_data[var] == 'Mannen', var] = 'Man'
            self.input_data.loc[self.input_data[var] == 'Vrouwen', var] = 'Vrouw'

        # Drop columns which are not used any further
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow', 'valid_from_min', 'valid_to_max', 'saldototaal_agg']
        list_to_drop = utils.doListIntersect(list_to_drop, self.input_data.columns) #ensures no error for a missing column
        self.input_data.drop(list_to_drop, axis = 1, inplace = True)

        self.input_data.sort_index(inplace = True, axis = 1) #Sort index

        print(f"Finished preparing data at {utils.get_time()}")

    def transform_for_cross_data( self, date_for_slice = "2019Q4" ):
        """
        Method to perform some transformations to prepare a cross-section.
        Selected variables are chosen to be take a mean over the year.

        In order to counter seasonal effects and large on-off values,
        this method uses information over the previous year and calculates
        a mean value over the year for selected variables (in search_list below)

        For several observations, the values for a certain period are missing.
        These are imputed by taking the next non-missing value and correcting it
        with an average index value for that period.

        The average index value for a period is calculated by first transforming every value to an index value
        which is index on the mean for a particular person. Then the average of these indices are taken over all persons
        to create a general index value for values in that period. This index value is then used to impute missing observations.

        :param date_for_slice: The period which is used to to create a cross section on.
        """
        #Create a list which contains part of the variables that will be taken into account to be averaged.
        search_list_counts = ['aantalatmtrans', 'aantalbetaaltrans', 'aantalfueltrans', 'aantallogins', 'aantalposttrans',
                              'aantaltegenrek', 'aantaltrans', 'logins_']
        exclusion_list_counts = ['bins']
        search_list_balance = ['saldototaal']
        exclusion_list_balance = []

        #Find variables which contain these string
        columns_to_use_list = utils.do_find_and_select_from_list(self.input_data.columns, search_list_counts,
                                                                 exclusion_list_counts)
        columns_to_use_list = columns_to_use_list + \
                              utils.do_find_and_select_from_list(self.input_data.columns, search_list_balance,
                                                                 exclusion_list_balance)
        indicators_list = ['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']
        columns_to_use_list = ['personid', 'period_obs'] + indicators_list + columns_to_use_list #columns to transform

        cross_df = self.input_data[columns_to_use_list].copy()
        time_conv_dict = {"Q": 4, "M": 12, "W": 52, "Y": 1} #Ensure that one year is taken for the analysis
        period_list = pd.period_range(end = date_for_slice, periods = time_conv_dict[self.current_freq], freq = self.current_freq)
        cross_df = cross_df[cross_df.period_obs.isin(period_list)]
        cross_df.sort_values(['period_obs', 'personid'], inplace = True) #Sort on period and personid in period

        #Find persons with incomplete data for one of the product segments
        incomplete_obs = cross_df.groupby('personid', as_index = False)[indicators_list].sum()
        incomplete_obs_person_bus = incomplete_obs.query("0 < has_bus_prtf < 4").personid
        incomplete_obs_person_jnt = incomplete_obs.query("0 < has_jnt_prtf < 4").personid
        incomplete_obs_person_ret = incomplete_obs.query("0 < has_ret_prtf < 4").personid

        #Take every personid which has a missing value for one of the segments
        incomplete_person_list = set(incomplete_obs_person_bus) | set(incomplete_obs_person_jnt) | set(incomplete_obs_person_ret)

        #Create a new dataset for incomplete persons to perform imputation on
        incomplete_index = cross_df.personid.isin(incomplete_person_list)
        incomplete_df = cross_df[incomplete_index]
        cross_df = cross_df[~incomplete_index]  # Select complete observations for now

        # Create mean values for every variable for every peson
        mean_values = cross_df.groupby('personid').mean()

        cross_df.set_index(["period_obs", "personid"], inplace = True)

        #Create an overall average index for every variable
        indexed_df = pd.DataFrame(columns = period_list, index = cross_df.columns)
        for period in period_list:
            indexed_result = cross_df.loc[period] / mean_values
            indexed_result = indexed_result[(indexed_result != np.inf) & (indexed_result != -np.inf)]
            indexed_df[period] = indexed_result.mean()
        indexed_df = indexed_df.transpose()
        cross_df.reset_index(inplace = True)

        ##LARGE OPERATION TO MATCH ALL COLUMNS WHICH ARE DEPENDENT ON THE SECTOR
        # business_columns = column_loop(cross_df, "business")
        business_columns = utils.do_find_and_select_from_list(cross_df.columns, ['business'])
        retail_columns = utils.do_find_and_select_from_list(cross_df.columns, ['retail'])
        joint_columns = utils.do_find_and_select_from_list(cross_df.columns, ['joint'])
        standard_cols = ['personid', 'period_obs']
        incomplete_final = incomplete_df[standard_cols]

        """-INTERPOLATE THE MISSING VALUES BASED ON THE GENERAL INDEX DERIVED ABOVE AND THE FIRST AVAILABLE QUANTITY 
        OF THE DIFFERENT PERSONIDS WITH MISSING OBSERVATIONS"""
        outer_merge_list = []
        for persons, cols, indic in [incomplete_obs_person_bus, business_columns, 'has_bus_prtf'], \
                                    [incomplete_obs_person_jnt, joint_columns, 'has_jnt_prtf'], \
                                    [incomplete_obs_person_ret, retail_columns, 'has_ret_prtf']:

            print(f"For '{indic}' , creating an interpolation of missing variables for the following columns: \n {cols} \n")

            incomplete_df_slice = incomplete_df[incomplete_df.personid.isin(persons)]
            # Get first available period value for each person
            incomplete_df_slice_persons = incomplete_df_slice.groupby(['personid'], as_index = False).apply(lambda x: x.loc[
                x[indic] == 1, 'period_obs'].min())
            #Rename to benchmark period and merge to create a final dataset containing personid and next available values
            # for imputation
            incomplete_df_slice_persons = incomplete_df_slice_persons.rename({None: "benchmark_period"}, axis = 1)
            incomplete_df_slice = pd.merge(incomplete_df, incomplete_df_slice_persons, on = 'personid')

            cols_without_parameters = standard_cols + ['benchmark_period', indic]
            cols_complete = cols_without_parameters + cols

            #Select certain values to use in the next part
            incomplete_df_slice = incomplete_df_slice[cols_complete]
            indexed_df_slice = indexed_df[cols]  # Take a slice of the variable that has previously been created to index

            inner_df_list = []
            outer_df_list = []

            #Period outer will test for each period if a personid has defnied value for the selected variables
            for period_outer in period_list:
                templist = (incomplete_df_slice[indic] == 0) & (incomplete_df_slice['period_obs'] == period_outer)

                # If a part has been defined in the period it will not be analyzed further and added to the end result.
                correct_list = (incomplete_df_slice[indic] == 1) & (incomplete_df_slice['period_obs'] == period_outer)
                outer_df = incomplete_df_slice.loc[correct_list, cols_complete]
                outer_df = outer_df.drop('benchmark_period', axis = 1)
                outer_df_list.append(outer_df) #List with already available data to merge later

                #If not defined, The value will be imputed.
                #Check what for each person will be the next available date and then for each date calculate
                # The newly imputed value. Inner df will be added to a list to concatenate later.
                for period_inner in period_list:
                    templist2 = templist & (incomplete_df_slice['benchmark_period'] == period_inner)
                    if templist2.sum() > 0:
                        templist3 = (incomplete_df_slice.personid.isin(incomplete_df_slice[templist2].personid)) & \
                                    (incomplete_df_slice.period_obs == period_inner)
                        inner_df = incomplete_df_slice.loc[templist3, cols] \
                                   * indexed_df_slice.loc[period_inner]
                        inner_df = pd.concat([incomplete_df_slice.loc[templist2, cols_without_parameters].reset_index(
                            drop = True), inner_df.reset_index(drop = True)], axis = 1, ignore_index = True)
                        inner_df.columns = cols_complete
                        inner_df = inner_df.drop("benchmark_period", axis = 1) #
                        inner_df_list.append(inner_df)

            #At the end create a larger dataset with imputed and non imputed values for the analysed variables
                        # And at this to overarching list to be used for concatenation later
            inner_and_outer = outer_df_list + inner_df_list
            outer_merge_list.append(pd.concat(inner_and_outer, ignore_index = True))

        #Concatenate the full end result for observations and variables in each object in the list
        for item in outer_merge_list:
            incomplete_final = pd.merge(incomplete_final, item, how = "left", on = standard_cols)
        incomplete_final.sort_index(axis = 1, inplace = True)

        #Fill with 0 for na values created by previous step
        incomplete_final[['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']] = incomplete_final[
            ['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']].fillna(value = 0)

        #Vars which have not been analysed and need to be merged with the new values
        remaining_vars = list(set(incomplete_df.columns) - set(incomplete_final.columns))
        remaining_vars = standard_cols + remaining_vars

        print(f"The final dataset has dim :{incomplete_final.shape} and the one before transforming has {incomplete_df.shape}")
        incomplete_final = pd.merge(incomplete_final, incomplete_df[remaining_vars], on = standard_cols)
        print(f"With final result : {incomplete_final.shape}. With indicator variables still in, which are discarded before the "
              f"final merge with the larger set.")

        # Concat to the larger set without NaN
        cross_df = pd.concat([cross_df, incomplete_final], ignore_index = True)
        cross_df.drop(indicators_list, axis = 1, inplace = True)
        cross_df = cross_df.groupby("personid").mean().reset_index()

        #Select variables to merge to the larger list and exclude if want to drop certain values
        # For a cross-section dataset
        remaining_vars = list(set(self.input_data.columns) - set(cross_df.columns))
        exclusion_list_total_vars = ['period_q']
        exclusion_list_total_vars = utils.do_find_and_select_from_list(self.input_data.columns, exclusion_list_total_vars)
        remaining_vars = list(set(['personid'] + remaining_vars) - set(exclusion_list_total_vars))
        print(f" Remaining vars in larger set:\n :{remaining_vars} \n \n after excluding the following variables :\n"
              f" {exclusion_list_total_vars}")

        # Takes last period and merges this to the variables who have just gotten a mean value
        partial_input_cross = self.input_data.loc[self.input_data.period_obs == period_list[-1], remaining_vars]
        self.transformed_df = pd.merge(cross_df, partial_input_cross, on = 'personid')
        print(f"Finished larger merge with final dimensions :{cross_df.shape}\n")

        print(f"Finished transforming data for cross section {utils.get_time()}")

    def add_longterm_change( self, benchmark_period ):
        """
        Takes value of certain dependent variables in benchmark_period and compares them to the current period.
        Also described as 'cross_long_df' and need observations in benchmark period and current period.
        :param benchmark_period: period to compare current values against
        :return:
        """
        """"
        Uses cross section, however does add variables which compare a certain historical point with
        the latest date.
        """
        product_counts_list = [
            'aantalproducten_totaal',
            'aantalproducten_totaal_business',
            'aantalproducten_totaal_joint',
            'aantalproducten_totaal_retail',
            'aantalproducten_accountoverlay',
            'portfolio_total_counts',
            'business_prtf_counts',
            'joint_prtf_counts',
            'retail_prtf_counts',
        ]
        #Create frequency value
        benchmark_period = pd.to_datetime(benchmark_period).to_period(self.current_freq)

        #Take values from this benchmark period and select relevant columns and personid
        benchmark_slice = self.input_data.query(f"period_obs == @benchmark_period")
        benchmark_slice = benchmark_slice[(['personid'] + product_counts_list)]

        #Index current period values
        transformed_df = self.cross_df[(['personid'] + product_counts_list)].copy()

        # Set personid as index for these new sets and sort the index
        transformed_df.set_index('personid', inplace = True)
        benchmark_slice.set_index('personid', inplace = True)
        transformed_df.sort_index(inplace = True)
        benchmark_slice.sort_index(inplace = True)

        #Create a delta value to see difference between current and benchmark period
        indicator_df1 = transformed_df - benchmark_slice
        #Create indicator value if has increased or not
        indicator_df2 = indicator_df1.where(indicator_df1 > 0, 0)
        indicator_df2 = indicator_df2.where(indicator_df1 == 0, 1)
        #Add prefix to variable names
        indicator_df1 = indicator_df1.add_prefix('delta_')
        indicator_df2 = indicator_df2.add_prefix('increased_')
        benchmark_slice = benchmark_slice.add_prefix('benchmark_')

        #Merge to larger transformed set
        df_to_merge = pd.concat([indicator_df1, indicator_df2, benchmark_slice], axis = 1)
        transformed_df.sort_index(axis = 1, inplace = True)
        self.transformed_df = pd.merge(self.transformed_df, df_to_merge, left_on = 'personid', right_index = True)
        pass

    def transform_for_panel( self ):
        """
        Calculates how much selected variables have changed and creates an indicator if this change is positive.
        First loops over the different time periods and creates change variables compared to the previous period.
        After that creates an indicator value where this value is positive
        Last step is merging it back to the larger self.prepared_df dataset
        """
        templist = [
            'personid',
            'period_obs',
            'aantalproducten_totaal',
            'aantalproducten_totaal_business',
            'aantalproducten_totaal_joint',
            'aantalproducten_totaal_retail',
            'aantalproducten_accountoverlay',
            'business_prtf_counts',
            'joint_prtf_counts',
            'retail_prtf_counts',
            'portfolio_total_counts'
        ]
        #Create new set to transform
        delta_df = self.input_data[templist].copy()
        delta_df = delta_df.set_index(['period_obs', 'personid'])
        delta_df.sort_index(inplace = True)

        #Loop over serveral periods and calculate change in values
        frame_list = []
        period_index = pd.period_range(start = self.first_date, end = self.last_date, freq = self.current_freq)
        for current_date_in_loop in period_index[1:]:
            new_delta_frame = delta_df.loc[current_date_in_loop, :] - delta_df.loc[current_date_in_loop - 1, :]
            new_delta_frame = new_delta_frame.reset_index()
            new_delta_frame['period_obs'] = current_date_in_loop
            frame_list.append(new_delta_frame)
        new_delta_frame = pd.concat(frame_list, ignore_index = True)

        #Create a list of columns which have been calculated and create indicator if value larger than one.
        templist = list(set(new_delta_frame.columns) - set(['period_obs', 'personid']))
        new_delta_frame[templist] = np.where(new_delta_frame[templist] > 1, 1, 0)

        #Add to larger dataset
        self.transformed_df = pd.merge(self.input_data, new_delta_frame, on = ['period_obs', 'personid'],
                                 suffixes = ["", "_delta"])

        print("number of positively changed variables is :\n", self.panel_df.iloc[:, -len(templist):].sum(), f"\nFrom a total of" \
                                                                                                             f" {self.panel_df.shape[0]} observations")
        print(f"Finished aggregating data for change in products at {utils.get_time()}")

    def load_split(self, dataset_basis = 'cross_df'):
        """
        Currently three datasets and estimations:
        Retail -> Has Joint or Not
        Retail (+values of Joint?) -> Has Business or not
        Retail + Business -> Has overlay or not (as every overlay has business connected)
        """
        assert dataset_basis in ['cross_df','cross_long_df','panel_df'], "Choose between 'cross_df'|'cross_long_df'|'panel_df|' "

        if self.imported_data_name != dataset_basis:
            self.transform_to_different_sets(transform_command = dataset_basis)

        self.transformed_df.sort_index(axis = 1, inplace = True)

        #Define string to be found in variable names to map them to a certain product segment
        general_exclusion_list = ['delta,aantalproducten,increased','prtf', 'dummy','change','benchmark']

        business_column_to_use = ['business','sbicode','sbiname','sbisector','aantal_types', 'saldototaal_fr','aantal_sbi',
                                  'aantal_sector']
        business_exclusion_list = [] + general_exclusion_list

        joint_column_to_use = ['joint']
        joint_exclusion_list = [] + general_exclusion_list

        retail_column_to_use = ['retail','educat4','age','child','hh_size','housetype','huidigewaarde','income']
        retail_exclusion_list =  ['businessage'] + general_exclusion_list

        #Find correct columns to use
        business_column_to_use = utils.do_find_and_select_from_list(self.transformed_df.columns, business_column_to_use,
                                                                    business_exclusion_list)
        joint_column_to_use = utils.do_find_and_select_from_list(self.transformed_df.columns, joint_column_to_use,
                                                                 joint_exclusion_list)
        retail_column_to_use = utils.do_find_and_select_from_list(self.transformed_df.columns, retail_column_to_use,
                                                                  retail_exclusion_list)

        drop_list_general = ['period','birthyear']
        drop_exclusion = []
        drop_list_general = utils.do_find_and_select_from_list(self.transformed_df, drop_list_general, drop_exclusion)
        ##Replace th

        #Select the relevant data needed for each data selection
        if self.current_data_selection == 'retail_only':
            columns_to_drop = business_column_to_use + joint_column_to_use + drop_list_general
            self.transformed_df = self.transformed_df.query("has_ret_prtf == 1")
            self.transformed_df.drop(columns_to_drop, axis = 1, inplace = True)

        elif self.current_data_selection in ['joint_and_retail','joint_or_retail']:
            columns_to_drop = business_column_to_use + drop_list_general
            self.transformed_df.drop(columns_to_drop, axis = 1, inplace = True)
            if self.current_data_selection == 'joint_and_retail':
                self.transformed_df = self.transformed_df.query(f"has_ret_prtf == 1 & has_jnt_prtf == 1")
            else:
                self.transformed_df = self.transformed_df.query(f"has_ret_prtf == 1 | has_jnt_prtf == 1")
        elif self.current_data_selection == 'retail_and_business':
            columns_to_drop = joint_column_to_use + drop_list_general
            self.transformed_df = self.transformed_df.query(f"has_ret_prtf == 1 & has_bus_prtf == 1", inplace = True)
            self.transformed_df.drop(columns_to_drop, axis = 1, inplace = True)

        self.transformation_log.append(f"{utils.print_seperated_list(columns_to_drop)} dropped for {self.current_data_selection}")
        print(self.transformation_log[-1])

        #Set loaded split to new split and add the list of variables to the general variables dict
        self.loaded_split = self.current_data_selection
        self.variables_dict = {'retail_column_to_use':retail_column_to_use,'joint_column_to_use':
            joint_column_to_use,
                               'business_column_to_use': business_column_to_use}

        print(f"Load split for '{self.current_data_selection}' has been loaded at {utils.get_time()}")

    def split_variable_sets( self, exclude_list_arg: list = []):
        """
        Splits the variables to different types of variables
        :param exclude_list_arg: string which to exclude when found in a variable
        :return:
        """
        if self.loaded_split != self.current_data_selection:
            self.load_split()

        if self.loaded_split == 'joint_or_retail':
            self.join_together_retail_joint()

        old_columns = self.transformed_df.columns
        self.transformed_df = self.transformed_df.dropna(axis = 1)
        self.transformation_log.append(f"Variables dropped for na values are "
                                       f" { utils.print_seperated_list(list(set(old_columns) - set(self.transformed_df.columns))) } ")
        print(self.transformation_log[-1])

        self.transformed_df.set_index('personid', inplace = True)
        dependent_search_list = ['has_','increased','prtf', 'benchmark', 'delta','aantalproducten','total_counts']
        exclude_general_list = [] + exclude_list_arg
        binary_exclusion_list = []
        exclude_categorical_list = ['total_counts']
        numerical_exclusion_list = []

        print(f"Selecting variables for {self.current_data_selection}")
        if self.current_data_selection == 'retail_only':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list
            numerical_exclusion_list = [] + numerical_exclusion_list + exclude_general_list

        elif self.current_data_selection in ['joint_and_retail','joint_or_retail']:
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list
            numerical_exclusion_list = [] + numerical_exclusion_list + exclude_general_list

        elif self.current_data_selection == 'retail_and_business':
            dependent_exclusion_list = []
            exclude_general_list = [] + exclude_general_list
            binary_exclusion_list = [] + binary_exclusion_list + exclude_general_list
            exclude_categorical_list = [] + exclude_categorical_list + exclude_general_list
            numerical_exclusion_list = [] + numerical_exclusion_list + exclude_general_list

        ##Selection of variables and
        dependent_variables = utils.do_find_and_select_from_list(self.transformed_df.columns, dependent_search_list,
                                                                 dependent_exclusion_list)
        self.transformation_log.append(f"The Following Variables are possible dependent variables (length"
                                 f" {len(dependent_variables)}): \n {dependent_variables}\n")
        print(self.transformation_log[-1])

        unique_vals = self.transformed_df.nunique()

        binary_variables = list(set(unique_vals[unique_vals <= 2].index) - set(dependent_variables) - set(binary_exclusion_list))
        binary_variables = utils.do_find_and_select_from_list(binary_variables,[""],binary_exclusion_list)
        self.transformation_log.append(f"The Following Variables have one value or are binary (length {len(binary_variables)}): \n"
            f" {binary_variables}\n")
        print(self.transformation_log[-1])


        categorical_variables = list(set(unique_vals[unique_vals <= 36].index))
        categorical_variables = list(set(categorical_variables) - set(dependent_variables) - set(binary_variables) - set(exclude_categorical_list))
        categorical_variables = utils.do_find_and_select_from_list(categorical_variables, [""], exclude_categorical_list)
        self.transformation_log.append(f"The Following Variables are categorical (length"
                                 f" {len(categorical_variables)}): \n {categorical_variables}\n")
        print(self.transformation_log[-1])


        numerical_variables = list(set(self.transformed_df.columns) - set(dependent_variables + binary_variables + categorical_variables))
        numerical_variables = utils.do_find_and_select_from_list(numerical_variables, [""], numerical_exclusion_list)
        self.transformation_log.append(f"The Following Variables are either float or integer  (lengt" \
                                    f"h {len(numerical_variables)}): \n {numerical_variables}\n")
        print(self.transformation_log[-1])

        total_variables = dependent_variables  + binary_variables + categorical_variables + numerical_variables
        templist = list(set(self.transformed_df.columns) - set(total_variables))
        self.transformation_log.append(f"Not categorized variables are (length {len(templist)}): \n {templist}\n")
        print(self.transformation_log[-1])

        self.loaded_variable_splits = self.current_data_selection
        self.variables_dict = {**self.variables_dict, 'dependent_variables':dependent_variables, 'binary_variables':binary_variables,
                            'categorical_variables':categorical_variables, 'numerical_variables':numerical_variables}
        self.transformed_df.sort_index(axis = 1, inplace = True)
        pass

    def join_together_retail_joint( self ):
        """
        If retail or joint is selected, will create dataset containing aggregation of retail and joint values
        to one value
        """
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

        unique_vals = self.transformed_df.nunique()
        binary_vals_list = ['sparenyn','depositoyn','betalenyn']
        intersected_no_binary = utils.do_find_and_select_from_list(intersected,[""],binary_vals_list)

        new_retail_vars = []
        new_joint_vars = []
        joint_retail_vars = []
        for item in intersected_no_binary:
            joint_retail_vars.append(f"{item}_retail_joint")
            new_retail_vars.append(f"{item}_retail")
            new_joint_vars.append((f"{item}_joint"))
            self.transformed_df[[new_retail_vars[-1], new_joint_vars[-1]]] = self.transformed_df[[
                new_retail_vars[-1],new_joint_vars[-1] ]].fillna(0,axis = 1)
            self.transformed_df[joint_retail_vars[-1]] = self.transformed_df[new_retail_vars[-1]] \
                                                         + self.transformed_df[new_joint_vars[-1]]

        intersected_binary = list( set(intersected) - set(intersected_no_binary) )
        for item in intersected_binary:
            joint_retail_vars.append(f"{item}_retail_joint")
            new_retail_vars.append(f"{item}_retail")
            new_joint_vars.append((f"{item}_joint"))
            self.transformed_df[[new_retail_vars[-1], new_joint_vars[-1]]]= self.transformed_df[[new_retail_vars[-1],
                                                                                                 new_joint_vars[-1]]].fillna(0,axis = 1)
            self.transformed_df[joint_retail_vars[-1]] = np.where((self.transformed_df[new_retail_vars[-1]] \
                                                                   + self.transformed_df[new_joint_vars[-1]]) > 0, 1, 0)

        exclude_drop_columns = ['aantalproducten','delta',]
        drop_columns = utils.do_find_and_select_from_list((new_joint_vars + new_retail_vars), [""],exclude_drop_columns)
        self.transformed_df = self.transformed_df.drop(drop_columns, axis = 1)
        pass

    def transform_variables( self, type_variables_to_transform = "all" ):
        """
        Transform variables for specific type of variable
        :param type_variables_to_transform: 'numerical_variables', 'categorical variables', 'all'
        :return:
        """
        if self.loaded_variable_splits != self.current_data_selection:
            self.split_variable_sets()

        #Transform numerical variables to new specifications
        if type_variables_to_transform in ['numerical_variables','all']:
            self.variables_dict['numerical_variables_with_dummies'] = []
            self.variables_dict['numerical_variables_discrete'] = []


            for variable in self.variables_dict['numerical_variables']:
                self.transform_numerical_variables(variable)

            self.value_var_transformed = True

        #Transform categorical values to new specifications
        #Makes a distrinction between ordered categorical values or unordered categories
        if type_variables_to_transform in ['categorical_variables','all']:
            unordered_search_string = ['finergy','geslacht']
            unordered_variables = utils.do_find_and_select_from_list(self.transformed_df.columns, unordered_search_string)

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
        """

        :param x_variable:
        :param n_splits:
        :return:
        """
        #Can add n_jobs = -1 to improve speed
        y_variable = self.current_dependent_var

        X = self.transformed_df.loc[:, x_variable].to_frame()
        y = self.transformed_df.loc[:, y_variable].to_frame()

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
            self.transformed_df[x_variable_discrete] = X['classified_to']
            self.transformed_df[x_variable_discrete] = self.transformed_df[x_variable_discrete] - 1
            minmax_for_discrete = X.groupby(by = 'classified_to', as_index = False)[x_variable].aggregate( ['min','max'] )
            minmax_for_discrete.sort_values('min', inplace = True)
            minmax_for_discrete.reset_index(drop = True, inplace = True)

            self.transformed_df[x_variable_discrete] = 0
            for i in range(0,minmax_for_discrete.shape[0]):
                change_var_index = (self.transformed_df[x_variable] >= minmax_for_discrete.loc[i, 'min']) & \
                                   (self.transformed_df[x_variable] <= minmax_for_discrete.loc[i, 'max'])
                self.transformed_df.loc[change_var_index, x_variable_discrete] = int(i)

            """"If more than two classes, additional dummy values are required for dummy specification"""
            if best_estimator.n_classes_ > 2:
                new_data = pd.get_dummies(self.transformed_df[x_variable_discrete], prefix = x_variable_discrete)
                self.variables_dict['numerical_variables_with_dummies'] =self.variables_dict['numerical_variables_with_dummies'] \
                                                                         + list(new_data.columns)

                self.transformation_log.append(f"Added dummies :{utils.print_seperated_list(new_data.columns)}")
                self.print_transformation_log()
                self.transformed_df = pd.concat([self.transformed_df, new_data], axis = 1)

        else:
            dropmessage = f"{x_variable}_discrete will not be created"
            self.transformation_log.append(dropmessage)
            self.print_transformation_log()

    def transform_categorical_variables(self , x_variable, threshold, is_unordered, max_cats = 5):
        """
        Transforms Categorical variables
        :param x_variable: variable to transform
        :param threshold: minimum number of observations in a
        :param is_unordered: iof ordered or unordered category
        :param max_cats: maximum number of categories to end up with
        :return:
        """
        y_variable = self.current_dependent_var
        X = self.transformed_df.loc[:, x_variable].to_frame()
        y = self.transformed_df.loc[:, y_variable].to_frame()
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
            UNORDERED VARIABLES - Creates a new value to map categories to which are too small for the threshold
            """

            new_data = pd.DataFrame(self.transformed_df[x_variable])
            if len(other_category) > 1:
                new_data.loc[X[x_variable].isin(other_category), x_variable] = "Other"
                self.transformed_df[x_variable_recoded] = new_data[x_variable]
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

                self.transformed_df[x_variable_recoded] = 0
                self.variables_dict['categorical_variables_recoded'].append(x_variable_recoded)
                self.transformation_log.append( f"{x_variable} has been recoded into {len(recoded_list)} categories" )
                self.print_transformation_log()

                transformation_log_add = ""
                for new_var in recoded_list:
                    recode_index = (self.transformed_df[x_variable] >= new_var[0]) & (self.transformed_df[x_variable] <= new_var[1])
                    self.transformed_df.loc[recode_index, x_variable_recoded] = new_var[2]
                    transformation_log_add = f"{transformation_log_add} || Lower bound:{new_var[0]} , Upper bound {new_var[1]}  has " \
                                             f"been coded with < {new_var[2]} > ||"

                self.transformation_log.append(transformation_log_add)
                self.print_transformation_log()
                has_been_recoded = True

        """" If a variable has been recoded """
        if has_been_recoded:
            new_data = pd.get_dummies(self.transformed_df[x_variable_recoded], prefix = f"{x_variable_recoded}")
            self.transformed_df = pd.concat([self.transformed_df, new_data], axis = 1)
            self.variables_dict['categorical_variables_recoded_dummies'] = self.variables_dict[
                                                                                 'categorical_variables_recoded_dummies'] \
                                                                         + list(new_data.columns)

        """FINAL CONCATENATION OF FILES"""
        new_data = self.transformed_df[x_variable].copy()
        new_data = pd.get_dummies(new_data, prefix = f"{x_variable}")
        self.transformed_df = pd.concat([self.transformed_df, new_data], axis = 1)
        self.variables_dict['categorical_variables_dummies'] = self.variables_dict['categorical_variables_dummies'] \
                                                                         + list(new_data.columns)
        self.transformation_log.append(f"Added {new_data.columns.shape[0]} dummy categories for each value of {x_variable}. "
                                       f"Category transformation finished\n")
        self.print_transformation_log()

        pass





        # =============================================================================
        # Methods for Data Processing
        # =============================================================================
        """
        Methods for data creation for general cross-section data and machine learning methods.
        To correctly run, just provide which transformation is warranted. It does not yet call upon the dataprocessor to 
        create the base data needed to use this algorithm however. 
        """



    def set_dates( self, first_date, last_date, override_folder_change = False ):
        "Method for safely changing dates in this class"
        data_freq = utils.infer_date_frequency(first_date)
        assert data_freq != None \
               and (utils.infer_date_frequency(last_date) != None or \
                    last_date == ""), \
            "No Valid date set"
        self.first_date = first_date
        self.current_freq = data_freq

        if last_date == "":
            self.last_date = first_date
            self.multiple_periods_imported = False
            print(f"Period set to {self.first_date} with frequency {self.current_freq}")
        else:
            self.last_date = last_date
            self.multiple_periods_imported = True
            print(f"Period set from {self.first_date} to {self.last_date} with frequency {self.current_freq}")

        if self.automatic_folder_change and not override_folder_change:
            self.folder_operations("replace_time", self.first_date, self.last_date)

    ###------LAST TRANSFORMATION TO DATA BEFORE MODELLING ----------###


    # =============================================================================
    # Methods for Formal Testing and Estimation
    # =============================================================================

    def test_general_importance_vars(self, n_to_use = 4):
        """

        :param n_to_use:
        :return:
        """
        y = self.transformed_df[self.current_dependent_var]

        new_seed =  174011
        new_seed2 = 871124

        #Create one kfold split to be used in every instance
        split_to_use = []
        for train, test in  StratifiedKFold(n_to_use, shuffle = True, random_state = new_seed).split(np.zeros(self.transformed_df.shape[
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
                x_selected = self.transformed_df[new_variables_to_use2]
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

    def run_hyperparameter_tuning( self, n_splits = 4):
        """
        Run the n_splits Cross Validation. Inner loop for tuning of hyperparameters and outer loop for training full results.
        :param n_splits: number of splits in stratified kfold
        :return:
        """
        """
        
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
        X = self.transformed_df[cols]
        y = self.transformed_df[self.current_dependent_var]
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
        """
        Runs final model and prints importances
        :return:
        """
        seed_new = 978391

        cols = self.variables_dict['binary_variables'] + self.variables_dict['final_variables']
        X = self.transformed_df[cols]
        y = self.transformed_df[self.current_dependent_var]

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

    # =============================================================================
    # General axcilliary methods
    # =============================================================================
    def import_data( self, import_string: str, first_date: str, last_date = "", addition_to_file_name = "" ):
        """
        Imports list of files created and exported from Knab_dataprocessor.
        :param import_string: type of files to be imported (see code to see names)
        :param first_date: first date of these files
        :param last_date: last_date to import of these files
        :param addition_to_file_name: if additional string in filename that has not
        been defined in the code below, could be added here
        :return:
        """
        if last_date != "":
            #Creates a larger file from the smaller files created by the dataprocessor
            import_list = []
            self.current_freq = utils.infer_date_frequency(first_date)
            for current_date_in_loop in pd.period_range(start = first_date, end = last_date, freq = self.current_freq):
                self.import_data(import_string, first_date = current_date_in_loop, addition_to_file_name = addition_to_file_name)
                self.last_imported['period_obs'] = current_date_in_loop
                import_list.append(self.last_imported.copy() )
            self.input_data = pd.concat(import_list, ignore_index = True)
        else:
            if import_string == 'input_cross':
                self.last_imported = pd.read_csv(f"{self.interdir}/final_df_{first_date}{addition_to_file_name}.csv")

    def folder_operations( self, folder_command, first_date = None, last_date = None, keywords_list = None, **extra_args ):
        """
        Performs some operations in folders created in analysis
        :param folder_command: what kind of operation should be performed
        :param first_date: first date of folder
        :param last_date: last date of folder
        :param keywords_list: keywords to search for in filenames
        :param extra_args:
        :return:
        """
        if (first_date == None) and (folder_command in ['create_sub_and_import', 'replace_time']):
            first_date = self.first_date
            last_date = self.last_date
        else:
            if last_date == None:
                last_date = first_date

        ##Importing different sets of data
        if keywords_list == None:
            keywords_list = ['final_df', 'valid_id', 'portfoliolink', 'base_experian', 'base_linkinfo']

        if folder_command == "create_sub_and_import":
            utils.create_subfolder_and_import_files(first_date = first_date, last_date = last_date, subfolder = self.interdir,
                                                    find_list = keywords_list, **extra_args)
        elif folder_command == "replace_time":
            utils.replace_time_period_folder(first_date = first_date, last_date = last_date, subfolder = self.interdir,
                                             remove_list = keywords_list, **extra_args)
        elif folder_command == 'clean_folder':
            utils.replace_time_period_folder(subfolder = self.interdir, remove_list = keywords_list, **extra_args)
        else:
            print("Wrong Value: Choose either |'final_df', 'create_sub_and_import','clean_folder' |")



    """
    IMPORTING DATA AND VARIABLE SETTERS
    """


    def print_transformation_log( self, print_total = False ):
        """
        Either prints the last addition to the transformation log or
        if print_total = True, the whole transformation log.
        :param print_total: Set True to print every entry in the log
        :return:
        """
        if print_total:
            for i,log_item in enumerate(self.transformation_log):
                print(f"Log entry {i+1} :\n")
                print(f"{log_item}\n \n")
        else:
            if self.do_print_logs:
                print(self.transformation_log[-1])

    def create_archive_folders( self ):
        """
        Create folder to archive latest result in
        :return:
        """
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
        """
        Save plots that are created
        :return:
        """
        if not self.archiving_active:
            self.create_archive_folders()
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.subarchive_dir}/plotsoutput.pdf")
        pdf.savefig(plt.gcf())
        pdf.close()

        plot_time_string = datetime.now().strftime("%m%d_%H%M%S")
        plotname = f"{self.plotsdir}/{plot_time_string}_plot.png"
        plt.savefig(plotname)

    def save_to_excel( self , data,addition_to_name = ""):
        """
        Save a dataframe to excel
        :param data: data to save
        :param addition_to_name: name to specify results
        :return:
        """
        if not self.archiving_active:
            self.create_archive_folders()

        time_string = datetime.now().strftime("%m%d_%H%M%S")
        log_filename = f"{self.subarchive_dir}/{time_string}_results{addition_to_name}.xlsx"
        data.to_excel(log_filename, index = False)


    def save_transformation_log( self ):
        """
        Save the transformation log as txt file
        :return:
        """
        if not self.archiving_active:
            self.create_archive_folders()

        time_string = datetime.now().strftime("%m%d_%H%M%S")
        log_filename = f"{self.subarchive_dir}/{time_string}_transformation_log.txt"

        f = open(log_filename, "x")
        for log_entry in self.transformation_log:
            f.write(log_entry)
        f.close()

    def force_error( self ):
        """"Used for debugging"""
        for item in dir():
            if (item[0:2] != "__") and (item[0] != "_"):
                del globals()[item]

    """"
    SETTERS AND GETTERS
    """

    def set_current_data_selection_to_use( self, data_selection_to_use):
        selectionnames = ['retail_only', 'joint_and_retail', 'joint_or_retail', 'retail_and_business']
        assert data_selection_to_use in selectionnames, f"choose a value from {selectionnames}"
        self.current_data_selection = data_selection_to_use

    def set_current_dependent_variable( self, dependent_variable_to_use ):
        if self.loaded_variable_splits != None:
            assert dependent_variable_to_use in self.transformed_df.columns, "Dependent variable not found in columns"
        self.current_dependent_var = dependent_variable_to_use

    def set_dates(self, first_date, last_date):
        "Method for safely changing dates in this class"
        data_freq = utils.infer_date_frequency(first_date)
        assert data_freq != None \
               and (utils.infer_date_frequency(last_date) != None or
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

    def get_predefined_var( self, return_num ):

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

        if return_num == 1:
            return final_variables_1

    def get_predefined_param(self, return_num: int):
        parameters_1 = {'criterion'   : 'gini', 'max_depth': 5, 'max_features': 17, 'min_samples_split': 20, 'n_estimators': 100,
                    'random_state': 978391}

        if return_num == 1:
            return parameters_1