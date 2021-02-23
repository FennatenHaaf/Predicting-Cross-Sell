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
        self.prepared_df = pd.DataFrame()

    def set_dates(self, first_date, last_date):
        "Method for safely changing dates in this class"
        assert utils.infer_date_frequency(first_date) != None \
               and (utils.infer_date_frequency(first_date) != None or\
               last_date == ""),\
            "No Valid date set"
        self.first_date = first_date

        if last_date == "":
            self.last_date = first_date
            self.multiple_periods_imported = False
            print(f"Period set to {self.first_date}")
        else:
            self.last_date = last_date
            self.multiple_periods_imported = True
            print(f"Period set from {self.first_date} to {self.last_date}")

    ###------LAST TRANSFORMATION TO DATA BEFORE MODELLING ----------###
    def prepare_data(self, first_date = "", last_date = ""):
        print(f"Starting preparing data at {utils.get_time()}")
        if self.first_date == None:
            if first_date == "":
                self.set_dates("2017Q3","2019Q4")
            else:
                self.set_dates(first_date,last_date)

        #Sort the index and make a deepcopy of the original data
        self.import_data('input_cross', first_date = self.first_date, last_date = self.last_date)

        for i, df in enumerate(self.input_cross_list):
            new_df = additionalDataProcess.aggregate_portfolio_types(df)
            self.input_cross_list[i] = df
        self.prepared_df = pd.concat(self.input_cross_list, ignore_index =  True)

        #Sort columns and make lowercase names
        self.prepared_df.rename(str.lower, axis = 'columns', inplace =  True)
        self.prepared_df.sort_index(inplace = True, axis = 1)

        #Drop unnecessary columns
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow','valid_from_min','valid_to_max']
        list_to_drop = utils.doListIntersect(list_to_drop, self.prepared_df.columns)
        self.prepared_df.drop(list_to_drop, axis = 1, inplace = True)
        self.prepared_df[['business', 'joint', 'retail']] = self.prepared_df[['business', 'joint', 'retail']].fillna(value = 0)

        rename_dict = {
            'business': 'has_bus_prtf',
            'retail'  : 'has_ret_prtf',
            'joint'   : 'has_joint_prtf'
        }

        rename_dict = utils.doDictIntersect(self.prepared_df.columns, rename_dict)
        self.prepared_df.rename(rename_dict,axis = 1, inplace = True)
        self.prepared_df = self.prepared_df.assign(
            period_q2 = lambda x: np.where(x.period_obs.dt.quarter == 2, 1, 0),
            period_q3 = lambda x: np.where(x.period_obs.dt.quarter == 3, 1, 0),
            period_q4 = lambda x: np.where(x.period_obs.dt.quarter == 4, 1, 0),
            has_bus_joint_prtf = lambda x: x.has_bus_prtf * x.has_joint_prtf,
            has_bus_ret_prtf=  lambda x: x.has_bus_prtf * x.has_joint_prtf,
            has_joint_ret_prtf = lambda x: x.has_bus_prtf * x.has_joint_prtf
        )


        self.transform_for_change_in_variables()
        self.transform_for_cross_data()
        print(f"Finished preparing data at {utils.get_time()}")

    def transform_for_change_in_variables( self ):
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
            'aantalproducten_totaal_retail'
        ]
        delta_df = self.prepared_df[templist].copy()
        delta_df = delta_df.set_index(['period_obs', 'personid'])
        delta_df.sort_index(inplace = True)

        date_freq = utils.infer_date_frequency(self.first_date)
        frame_list = []
        for current_date_in_loop in pd.period_range(start = self.first_date, end = self.last_date, freq = date_freq)[1:]:
            new_delta_frame = delta_df.loc[current_date_in_loop,] - delta_df.loc[current_date_in_loop - 1]
            new_delta_frame = new_delta_frame.reset_index()
            new_delta_frame['period_obs'] = current_date_in_loop
            frame_list.append(new_delta_frame)
        new_delta_frame = pd.concat(frame_list, ignore_index = True)

        templist = list(set(new_delta_frame.columns) - set(['period_obs', 'personid']))
        new_delta_frame[templist] = np.where(new_delta_frame[templist] > 1, 1, 0)

        self.prepared_df = pd.merge(self.prepared_df, new_delta_frame, on = ['period_obs','personid'], suffixes = ["","_delta"])
        print(f"Finished aggregating data for change in products at {utils.get_time()}")

    def transform_for_cross_data(self):
        # if self.multiple_periods_imported:
        #     self.aggregate_to_year()
        # def aggregate_to_year(self):
        #     "Method to aggregate data to a year"
        #     # Todo Aggregate values to year
        #
        #     self.prepared_df['year_obs'] = self.prepared_df.period_obs.dt.year
        #     self.prepared_df.set_index(['year_obs','personid','period_obs'], inplace = True)
        #
        #     current_year = 2018
        #     #Counts
        #
        #     #Categoricals
        #
        #
        #     #Balance
        #
        #     templist = utils.doListIntersect(declarationsFile.get_cross_section_aggregation("balance"), self.prepared_df.columns)
        #     balance_set = self.prepared_df.loc[2018,templist]
        #
        #
        #     balance_part = self.prepared_df[templist]
        #
        #
        #
        #     #YesNo
        #
        #
        #     #Last Part
        print(f"Finished aggregating data for cross data analysis at {utils.get_time()}")


    #TODO Add several dummy variables
    #TODO Reduce the number of categories by either selection or perhaps PCA/FA.
        pass


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
            date_freq = utils.infer_date_frequency(first_date)
            for current_date_in_loop in pd.period_range(start= first_date, end= last_date, freq= date_freq):
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