"""
Contains extra functions to transform data

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import utils
import dataInsight
import declarationsFile
import gc
from tqdm import tqdm
from os import path
import re


def get_difference_data(this_period,prev_period):
    """Get the datapoints for which there is a difference of 1 or more
    portfolios"""
    
    # Get the total number of portfolios in each period
    this_period['portfoliototaal']= this_period[["business","retail",
                                        "joint"]].sum(axis=1)
    prev_period['portfoliototaal']= prev_period[["business","retail",
                                        "joint"]].sum(axis=1)
    #TODO: may also want to add bookkeeping overlay? maybe not s
    
    # Get saldo difference
    this_period['saldototaal'] = this_period[["saldototaal_business",
                                              "saldototaal_retail",
                                              "saldototaal_joint"]].sum(axis=1)
    prev_period['saldototaal'] = prev_period[["saldototaal_business",
                                              "saldototaal_retail",
                                              "saldototaal_joint"]].sum(axis=1)
    this_period['percdiff'] =((this_period['saldototaal'] \
                                - prev_period['saldototaal']) / prev_period['saldototaal'] )*100

    # Get portfolio variables
    for name in (["business","retail","joint"]):
        this_period[f"{name}_change"] = this_period[name] - prev_period[name]

    # TODO could also first transform saldototaal to log and then just take 
    # differences? -> what if a change is negative?

    select = (this_period['portfoliototaal']>prev_period['portfoliototaal'])
    data = this_period.loc[select,'percdiff']
    
    # Now add a variable for which type of data was added
    
    # Variables to add: type of portfolios added, characteristics current, 
    # saldo of the individual types??
    
    # Do a for loop outside this and add the thing to a df
    
    return data
    
    