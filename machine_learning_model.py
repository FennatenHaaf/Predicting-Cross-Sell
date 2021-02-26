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

class MachineLearningModel(object):

    def __init__(self, indir, interdir,outdir, automatic_folder_change = False):
        """
        Initialize the method to create a Cross Section Model and To model.
        """


        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir
        self.automatic_folder_change = automatic_folder_change
        self.seed = 978391 #Fixed seed
        ###Current default Base year###
        self.first_date = None

        #Empty lists to parse if non existant
        self.cross_long_df = None
        self.panel_df = None

    def test_static_features(self):
        if self.cross_long_df == None:
            self.import_data_to_model(import_command = "cross_long_df")

        self.cross_long_df.sort_index(axis = 1, inplace = True)

        retail_cl_df = self.cross_long_df[self.cross_long_df.has_ret_prtf == 1].copy()
        business_cl_df = self.cross_long_df[self.cross_long_df.has_bus_prtf == 1].copy()
        joint_cl_df = self.cross_long_df[self.cross_long_df.has_jnt_prtf == 1].copy()

        retail_cl_df.dropna(axis = 1, inplace = True)
        business_cl_df.dropna(axis = 1, inplace = True)
        joint_cl_df.dropna(axis = 1, inplace = True)
        ##To quickly get variables which are of a certain sector
        # retail_nas = retail_cl_df.isna().sum()
        # retail_nas = retail_nas[retail_nas > 0].index
        # business_nas = business_cl_df.isna().sum()
        # business_nas = business_nas[business_nas > 0].index


        pass

    def test_dynamic_features(self):
        if self.panel_df == None:
            self.import_data_to_model(import_command = "panel_df")

        pass

    ##IMPORTING DATA AND VARIABLE SETTERS
    def import_data_to_model(self, import_command, last_date = "", first_date = ""):
        process_data = additionalDataProcess.AdditionalDataProcess(self.indir,self.interdir,self.outdir)
        process_data.automatic_folder_change = self.automatic_folder_change
        if import_command == "cross_long_df":
            process_data.transform_to_different_sets("cross_long_df", first_date = first_date, last_date = last_date)
            self.cross_long_df = process_data.cross_long_df
        if import_command == "panel_df":
            process_data.transform_to_different_sets("panel_df", first_date = first_date, last_date = last_date)
            self.panel_df = process_data.panel_df

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
