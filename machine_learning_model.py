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

    def load_split(self, split_to_load):
        """
        Currently three datasets and estimations:
        Retail -> Has Joint or Not
        Retail (+values of Joint?) -> Has Business or not
        Retail + Business -> Has overlay or not (as every overlay has business connected)
        """
        if self.cross_long_df == None:
            self.import_data_to_model(import_command = "cross_long_df")

        self.cross_long_df.sort_index(axis = 1, inplace = True)

        #TODO create loop to get correct column values
        business_column_to_use = ['business','sbicode','sbiname','sbisector','aantal_types', 'saldototaal_fr','aantal_sbi',
                                  'aantal_sector']
        business_exclusion_list = ['dummy','change','increase','prtf','increased','delta']

        joint_column_to_use = ['joint']
        joint_exclusion_list = ['dummy','change','increase']

        retail_column_to_use = ['retail','educat4','age']
        retail_exclusion_list =  ['businessage','benchmark','increase','']

        #Find correct columns to use
        business_column_to_use = utils.do_find_and_select_from_list(self.cross_long_df.columns,business_column_to_use,business_exclusion_list)
        joint_column_to_use = utils.do_find_and_select_from_list(self.cross_long_df.columns, joint_column_to_use,joint_exclusion_list)
        retail_column_to_use = utils.do_find_and_select_from_list(self.cross_long_df.columns, retail_column_to_use,retail_exclusion_list)

        drop_list_extra = ['period_q']
        drop_exclusion = []
        drop_list_general = utils.do_find_and_select_from_list(self.cross_long_df,drop_list_extra,drop_exclusion)
        ##Replace th

        if split_to_load == 'retail_only':
            columns_to_drop = business_column_to_use + joint_column_to_use + drop_list_general
            self.train_set = self.cross_long_df.query("has_ret_prtf == 1").drop(columns_to_drop,
                                                                                axis = 1)
            #TODO Zoek waar verkeerde waarden vandaan komen voor de laatste periode
            self.train_set.dropna(inplace = True)

        elif split_to_load == 'joint_and_retail':
            self.train_set = self.cross_long_df.query(f"has_ret_prtf == 1 & has_jnt_prtf == 1").drop(business_column_to_use,
                                                                                                     axis = 1)
        elif split_to_load == 'retail_and_business':
            self.train_set = self.cross_long_df.query(f"has_ret_prtf == 1 & has_bus_prtf == 1").copy()

        unique_vals = self.train_set.nunique()
        templist = list(unique_vals[unique_vals <= 1].index)
        print(f"dropping variables {templist} for having one value")
        self.train_set.drop(templist, inplace = True)

        pass

    def test_static_features(self):
        self.load_split('retail_only')

        dependent_variable_increase = ['increased_business_prtf_counts']
        unique_vals = self.train_set.nunique()
        templist = list(set(unique_vals[unique_vals <= 2].index))
        print()

        templist = list(unique_vals[unique_vals <= 5].index)
        print(templist)

    def split_by_decision_tree(self,x,y):
        DecisionTreeClassifier()


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
