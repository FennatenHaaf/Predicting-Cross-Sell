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

import re
import importlib

class CrossSectionModels(object):

    def __init__(self, indir, interdir,outdir):
        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir

        #Empty lists to parse if non existant
        self.input_cross = pd.DataFrame()
        self.prepared_df = pd.DataFrame()

    ###------LAST TRANSFORMATION TO DATA BEFORE MODELLING ----------###
    def prepare_data(self):
        #Sort the index and make a deepcopy of the original data
        self.prepared_df = self.input_cross.sort_index( axis=1, key=lambda x: x.str.lower() ).copy()

        #Drop unnecessary columns and put personid as index
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow',]
        self.prepared_df = utils.check_and_drop_columns(self.prepared_df,list_to_drop)
        self.prepared_df.set_index('personid', inplace = True)



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
            for current_date_in_loop in pd.date_range(start= first_date, end= last_date, freq= date_freq):
                self.import_data(import_string, first_date = current_date_in_loop, addition_to_file_name = addition_to_file_name)
                exec( f"list_of_date_files.append(self.{import_string})" )
            exec(f"self.{import_string} = pd.concat(list_of_date_files, ignore_index= True)")
            pass

        if import_string == 'input_cross':
            self.input_cross = pd.read_csv(f"{self.interdir}/final_df_{first_date}{addition_to_file_name}.csv")


    def export_data(self):
        pass

    def debug_in_class(self):
        pass