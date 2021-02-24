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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

import re
import importlib
import gc

class CrossSectionModels(object):

    def __init__(self, indir, interdir,outdir,):
        """
        Initialize the method to create a Cross Section Model and To model.
        """


        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir

        ###Current default Base year###
        self.first_date = None

        #Empty lists to parse if non existant
        self.input_cross = pd.DataFrame()
        self.input_cross_df = pd.DataFrame()
        self.panel_chained_df = pd.DataFrame()
        self.cross_df = pd.DataFrame()
        self.cross_compared_df = pd.DataFrame()

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
            self.multiple_periods_imported = False
            print(f"Period set to {self.first_date} with frequency {self.current_freq}")
        else:
            self.last_date = last_date
            self.multiple_periods_imported = True
            print(f"Period set from {self.first_date} to {self.last_date} with frequency {self.current_freq}")

    ###------LAST TRANSFORMATION TO DATA BEFORE MODELLING ----------###
    def prepare_before_transform( self, first_date = "", last_date = "" ):
        print(f"Starting preparing data at {utils.get_time()}")
        if first_date == "":
            self.set_dates("2018Q1","2020Q4")
        else:
            self.set_dates(first_date,last_date)

        #Sort the index and make a deepcopy of the original data
        self.import_data('input_cross', first_date = self.first_date, last_date = self.last_date)

        for i, df in enumerate(self.input_cross_list):
            new_df = additionalDataProcess.aggregate_portfolio_types(df)
            self.input_cross_list[i] = df
        self.input_cross_df = pd.concat(self.input_cross_list, ignore_index =  True)

        #Make lowercase names
        self.input_cross_df.rename(str.lower, axis = 'columns', inplace =  True)

        #Drop unnecessary columns
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow','valid_from_min','valid_to_max']
        list_to_drop = utils.doListIntersect(list_to_drop, self.input_cross_df.columns)
        self.input_cross_df.drop(list_to_drop, axis = 1, inplace = True)
        self.input_cross_df[['business', 'joint', 'retail', 'accountoverlay']] = self.input_cross_df[['business', 'joint',
                                                                                                 'retail', 'accountoverlay']].fillna(value = 0)
        rename_dict = {
            'retail'  : 'has_ret_prtf',
            'joint'   : 'has_jnt_prtf',
            'business': 'experian_business'
        }

        rename_dict = utils.doDictIntersect(self.input_cross_df.columns, rename_dict)
        self.input_cross_df.rename(rename_dict, axis = 1, inplace = True)
        self.input_cross_df = self.input_cross_df.assign(
            period_q2 = lambda x: np.where(x.period_obs.dt.quarter == 2, 1, 0),
            period_q3 = lambda x: np.where(x.period_obs.dt.quarter == 3, 1, 0),
            period_q4 = lambda x: np.where(x.period_obs.dt.quarter == 4, 1, 0),
            has_bus_prtf = lambda x: np.where(x.aantalproducten_totaal_business > 0, 1, 0),
            has_bus_jnt_prtf = lambda x: x.has_bus_prtf * x.has_jnt_prtf,
            has_bus_ret_prtf=  lambda x: x.has_bus_prtf * x.has_ret_prtf,
            has_jnt_ret_prtf = lambda x: x.has_ret_prtf * x.has_jnt_prtf
        )
        self.input_cross_df.sort_index(inplace = True, axis = 1)

    def transform_to_different_sets(self, transform_command = "all"):

        if transform_command in ['cross','all']:
            self.prepare_before_transform("2019Q1","2019Q4")
            self.transform_for_cross_data()

        if transform_command in ['panel_chained', 'all']:
            self.prepare_before_transform("2017Q3", "2019Q4")
            self.transform_for_chained_change_in_variables()


        print(f"Finished preparing data at {utils.get_time()}")
    def transform_for_cross_data(self,date_for_slice = "2019Q4"):
        columns_to_use_list = utils.doListIntersect(self.input_cross_df.columns, declarationsFile.get_cross_section_agg('count_list'))
        columns_to_use_list = columns_to_use_list + \
                              utils.doListIntersect(self.input_cross_df.columns, declarationsFile.get_cross_section_agg(
                                  'balance_at_moment'))
        indicators_list = ['has_bus_prtf','has_jnt_prtf','has_ret_prtf']
        columns_to_use_list = ['personid', 'period_obs'] + indicators_list + columns_to_use_list

        cross_df = self.input_cross_df[columns_to_use_list].copy()
        time_conv_dict = {"Q":4,"M":12,"W":52,"Y":1}
        period_list = pd.period_range(end =  date_for_slice,periods = time_conv_dict[self.current_freq], freq = self.current_freq)
        cross_df = cross_df[cross_df.period_obs.isin(period_list)]
        cross_df.sort_values(['period_obs','personid'], inplace =  True)

        incomplete_obs = cross_df.groupby('personid', as_index =  False)[indicators_list].sum()
        incomplete_obs_person_bus = incomplete_obs.query("0 < has_bus_prtf < 4").personid
        incomplete_obs_person_jnt = incomplete_obs.query("0 < has_jnt_prtf < 4").personid
        incomplete_obs_person_ret = incomplete_obs.query("0 < has_ret_prtf < 4").personid

        incomplete_person_list = set(incomplete_obs_person_bus) | set(incomplete_obs_person_jnt) | set(incomplete_obs_person_ret)

        incomplete_index = cross_df.personid.isin(incomplete_person_list)
        incomplete_df = cross_df[incomplete_index]
        cross_df = cross_df[~incomplete_index] #Select complete observations for now

        mean_values = cross_df.groupby('personid').mean()
        cross_df.set_index(["period_obs","personid"], inplace = True)

        indexed_df = pd.DataFrame(columns = period_list, index = cross_df.columns)
        for period in period_list:
            indexed_result = cross_df.loc[period] / mean_values
            indexed_df[period] = indexed_result.mean()
        indexed_df = indexed_df.transpose()
        cross_df.reset_index(inplace = True)

        def column_loop(data: pd.DataFrame(),value_to_find: str):
            column_list = []
            for item in data.columns:
                if value_to_find in item:
                    column_list.append(item)
            return column_list

        ##LARGE OPERATION TO MATCH ALL COLUMNS WHICH ARE DEPENDENT ON THE SECTOR
        business_columns = column_loop(cross_df,"business")
        retail_columns = column_loop(cross_df,"retail")
        joint_columns = column_loop(cross_df,"joint")
        standard_cols = ['personid', 'period_obs']
        other_cols = set(incomplete_df.columns) - set(business_columns) - set(joint_columns) - set(standard_cols)
        incomplete_final = incomplete_df[standard_cols]

        """-INTERPOLATE THE MISSING VALUES BASED ON THE GENERAL INDEX DERIVED ABOVE AND THE FIRST AVAILABLE QUANTITY 
        OF THE DIFFERENT PERSONIDS WITH MISSING OBSERVATIONS"""
        outer_merge_list = []
        for persons,cols, indic in [incomplete_obs_person_bus,business_columns,'has_bus_prtf'],\
                   [incomplete_obs_person_jnt,joint_columns,'has_jnt_prtf'],\
                   [incomplete_obs_person_ret,retail_columns,'has_ret_prtf']:

            incomplete_df_slice = incomplete_df[incomplete_df.personid.isin(persons)]
            incomplete_df_slice_persons = incomplete_df_slice.groupby(['personid'], as_index = False).apply(lambda x: x.loc[
                x[indic] == 1,'period_obs'].min())
            incomplete_df_slice_persons = incomplete_df_slice_persons.rename({None: "benchmark_period"}, axis = 1)
            incomplete_df_slice = pd.merge(incomplete_df,incomplete_df_slice_persons, on = 'personid')


            cols_without_parameters = standard_cols + ['benchmark_period', indic]
            cols_complete = cols_without_parameters + cols

            incomplete_df_slice = incomplete_df_slice[cols_complete]
            indexed_df_slice = indexed_df[cols] #Take a slice of the variable that has previously been created to index

            inner_df_list = []
            outer_df_list = []

            for period_outer in period_list:
                templist = (incomplete_df_slice[indic] == 0) & (incomplete_df_slice['period_obs'] == period_outer)

                #If a part has been defined in the period it will not be analyzed further and added to the end result.
                correct_list = (incomplete_df_slice[indic] == 1) & (incomplete_df_slice['period_obs'] == period_outer)
                outer_df = incomplete_df_slice.loc[correct_list,cols_complete]
                outer_df = outer_df.drop('benchmark_period', axis = 1)
                outer_df_list.append(outer_df)

                for period_inner in period_list:
                    templist2 = templist & (incomplete_df_slice['benchmark_period'] == period_inner)
                    if templist2.sum() > 0:
                        templist3 = (incomplete_df_slice.personid.isin(incomplete_df_slice[templist2].personid) ) & \
                                    (incomplete_df_slice.period_obs == period_inner)
                        inner_df = incomplete_df_slice.loc[templist3,cols]\
                                                                       * indexed_df_slice.loc[period_inner]
                        inner_df = pd.concat([incomplete_df_slice.loc[templist2, cols_without_parameters].reset_index(
                            drop = True), inner_df.reset_index(drop = True)],axis = 1,ignore_index = True)
                        inner_df.columns = cols_complete
                        inner_df = inner_df.drop("benchmark_period", axis = 1)
                        inner_df_list.append(inner_df)

            inner_and_outer = outer_df_list + inner_df_list
            outer_merge_list.append(pd.concat(inner_and_outer, ignore_index =  True))

        for item in outer_merge_list:
            incomplete_final = pd.merge(incomplete_final, item, how = "left", on = standard_cols)
        incomplete_final.sort_index(axis = 1, inplace = True)

        incomplete_final[['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']] = incomplete_final[
            ['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']].fillna(value = 0)

        remaining_vars = list(set(incomplete_df.columns) - set(incomplete_final.columns))
        remaining_vars = standard_cols + remaining_vars


        print(f"The final dataset has dim :{incomplete_final.shape} and the one before transforming has {incomplete_df.shape}")
        incomplete_final = pd.merge(incomplete_final,incomplete_df[remaining_vars], on = standard_cols)
        print(f"With final result : {incomplete_final.shape}")

        #Concat to the larger set without NaN
        cross_df = pd.concat([cross_df,incomplete_final], ignore_index = True)

        #Clean variables
        del incomplete_df, incomplete_df_slice, incomplete_df_slice_persons, incomplete_final, incomplete_index,\
            incomplete_obs,incomplete_obs_person_ret, incomplete_obs_person_jnt, incomplete_obs_person_bus, incomplete_person_list,\
            indexed_df_slice,inner_df, inner_df_list, inner_and_outer, outer_merge_list, outer_df, outer_df_list, period_inner,\
            period_outer, templist, templist2, templist3
        gc.collect()

        cross_df = cross_df.groupby("personid").mean().reset_index()
        remaining_vars = list(set(self.input_cross_df) - set(cross_df))
        remaining_vars = ['personid'] + remaining_vars
        partial_input_cross = self.input_cross_df.loc[self.input_cross_df.period_obs == period_list[-1],remaining_vars]
        self.cross_df = pd.merge(cross_df,partial_input_cross, on = 'personid')

        print(f"Finished transforming data for cross section {utils.get_time()}")

    def transform_for_chained_change_in_variables( self ):
        """
        Calculates how much selected variables have changed and creates an indicator if this change is positive.
        First loops over the different time periods and creates change variables compared to the previous period.
        After that creates an indicator value where this value is positive
        Last step is merging it back to the larger self.prepared_df dataset
        """
        templist = [
            'personid','period_obs','aantalproducten_totaal',
            'aantalproducten_totaal_business',
            'aantalproducten_totaal_joint',
            'aantalproducten_totaal_retail',
            'accountoverlay'

        ]
        delta_df = self.input_cross_df[templist].copy()
        delta_df = delta_df.set_index(['period_obs', 'personid'])
        delta_df.sort_index(inplace = True)

        frame_list = []
        period_index = pd.period_range(start = self.first_date, end = self.last_date, freq = self.current_freq)
        for current_date_in_loop in period_index[1:]:
            new_delta_frame = delta_df.loc[current_date_in_loop,] - delta_df.loc[current_date_in_loop - 1]
            new_delta_frame = new_delta_frame.reset_index()
            new_delta_frame['period_obs'] = current_date_in_loop
            frame_list.append(new_delta_frame)
        new_delta_frame = pd.concat(frame_list, ignore_index = True)

        templist = list(set(new_delta_frame.columns) - set(['period_obs', 'personid']))
        new_delta_frame[templist] = np.where(new_delta_frame[templist] > 1, 1, 0)

        self.panel_chained_df = pd.merge(self.input_cross_df, new_delta_frame, on = ['period_obs', 'personid'], suffixes = ["", "_delta"])
        #Todo change period to +1
        print("number of positively changed variables is :\n", self.panel_chained_df.iloc[:, -4:].sum(), f"\nFrom a total of" \
                                                                                         f" {self.panel_chained_df.shape[0]} observations")
        print(f"Finished aggregating data for change in products at {utils.get_time()}")

    ###-----------------------MODELLING -----------------------------###
    def parameter_tuning(self):
        pass

    def compare_models(self):
        pass

    def train_and_predict_final_model(self):
        pass


    ###-----------------IMPORTING AND EXPORTING DATA----------------###
    def import_data(self,import_string: str,first_date: str,last_date = "" , addition_to_file_name = "" ):

        if last_date != "":
            exec(f"self.{import_string}_list = []")
            self.current_freq = utils.infer_date_frequency(first_date)
            for current_date_in_loop in pd.period_range(start= first_date, end= last_date, freq= self.current_freq):
                self.import_data(import_string, first_date = current_date_in_loop, addition_to_file_name = addition_to_file_name)
                exec(f"{import_string}_{current_date_in_loop} = self.{import_string}.copy()")
                exec(f"{import_string}_{current_date_in_loop}['period_obs'] = current_date_in_loop" )
                exec( f"self.{import_string}_list.append({import_string}_{current_date_in_loop})" )

        else:
            if import_string == 'input_cross':
                self.input_cross = pd.read_csv(f"{self.interdir}/final_df_{first_date}{addition_to_file_name}.csv")


    def export_data(self):
        pass

    def debug_in_class(self):
        "Method to be able to perform operations as if debugging in class method"
        print("hey")