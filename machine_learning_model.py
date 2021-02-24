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



    ###-----------------------MODELLING -----------------------------###
    def parameter_tuning(self):
        pass

    def compare_models(self):
        pass

    def train_and_predict_final_model(self):
        pass
