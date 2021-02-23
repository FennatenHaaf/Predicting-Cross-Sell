"""
Class to transform data and perform analysis on a cross-section of Knab Data.
"""
import utils
import declarationsFile
import dataInsight

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
        if self.first_date == None:
            if first_date == "":
                self.set_dates("2018Q1","2018Q4")
            else:
                self.set_dates(first_date,last_date)

        #Sort the index and make a deepcopy of the original data
        self.import_data('input_cross', first_date = self.first_date, last_date = self.last_date)
        self.prepared_df = self.input_cross.rename(str.lower, axis = 'columns')
        self.prepared_df.sort_index(inplace = True, axis = 1)

        #Drop unnecessary columns
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow',]
        list_to_drop = utils.doListIntersect(list_to_drop, self.prepared_df.columns)
        self.prepared_df.drop(list_to_drop, axis = 1, inplace = True)

        if self.multiple_periods_imported:
            self.aggregate_to_year()

    def aggregate_to_year(self):
        "Method to aggregate data to a year"
        # Todo Aggregate values to year
        #Counts

        #Categoricals

        #Balance

        #YesNo
        templist = utils.doListIntersect(declarationsFile.get_cross_section_aggregation("balance"), self.prepared_df.columns)
        balance_part = self.prepared_df[templist]

        #Last Part

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
            list_of_date_files = []
            date_freq = utils.infer_date_frequency(first_date)
            for current_date_in_loop in pd.period_range(start= first_date, end= last_date, freq= date_freq):
                self.import_data(import_string, first_date = current_date_in_loop, addition_to_file_name = addition_to_file_name)
                exec(f"{import_string}_{current_date_in_loop} = self.{import_string}.copy()")
                exec(f"{import_string}_{current_date_in_loop}['period_obs'] = current_date_in_loop" )

                exec( f"list_of_date_files.append({import_string}_{current_date_in_loop})" )
            exec(f"self.{import_string} = pd.concat(list_of_date_files, ignore_index= True)")

        else:
            if import_string == 'input_cross':
                self.input_cross = pd.read_csv(f"{self.interdir}/final_df_{first_date}{addition_to_file_name}.csv")


    def export_data(self):
        pass

    def debug_in_class(self):
        "Method to be able to perform operations as if debugging in class method"
        print("hey")