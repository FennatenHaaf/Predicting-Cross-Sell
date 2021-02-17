"""
Contains extra functions to transform data

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from datetime import date
import utils
import dataInsight
import declarationsFile
import gc
from tqdm import tqdm
from os import path
import re


# =============================================================================
# Methods for making saldo prediction dataset
# =============================================================================

def create_saldo_data(dflist, interdir, filename ="saldodiff", 
                      select_variables = None ):
    """Make saldo data out of a df list"""
    i=0
    diffdata = pd.DataFrame()
    
    for df in dflist:
        if i==0:    
            # We aggregate but don't transform the data
            dfold = aggregate_portfolio_types(df)
        else:
            dfnew = aggregate_portfolio_types(df)
            data = get_difference_data(dfnew,dfold,select_variables)
            diffdata = pd.concat([diffdata,data])
            dfold = dfnew
        i+=1

    print(f"Done creating saldo prediction dataset at {utils.get_time()}" \
          f", saving to {filename}")
        
    utils.save_df_to_csv(diffdata, interdir, filename, add_time = False )
    return diffdata


def get_difference_data(this_period,prev_period, log =True,
                        select_variables =None):
    """Get the datapoints for which there is a difference of 1 or more
    portfolios"""
    #TODO also add account overlay somehow so that we can also find the impact
    # of that (if there is any??)
    
    # Get the total number of portfolios in each period (excluding overlay)
    this_period['portfoliototaal']= this_period[["business","retail",
                                        "joint"]].sum(axis=1)
    prev_period['portfoliototaal']= prev_period[["business","retail",
                                        "joint"]].sum(axis=1)

    # Get saldo difference
    # this_period['saldototaal'] = this_period[["saldototaal_business",
    #                                           "saldototaal_retail",
    #                                           "saldototaal_joint"]].sum(axis=1)
    # prev_period['saldototaal'] = prev_period[["saldototaal_business",
    #                                           "saldototaal_retail",
    #                                           "saldototaal_joint"]].sum(axis=1)
    # Not necessary, we have already aggregated the data
    
    
    if log:
        # We take the log difference
        this_period['percdiff'] = getlogs(this_period['saldototaal']) \
                                   - getlogs(prev_period['saldototaal'])
        #this_period['percdiff'] = (np.log(this_period['saldototaal']+1)) \
        #                            - np.log((prev_period['saldototaal']+1))
    else:
         # Take the percentage - we add +1 to deal with 0 in the data
         this_period['percdiff'] =(((this_period['saldototaal']+1) \
                                - (prev_period['saldototaal']+1)) / (prev_period['saldototaal']+1) )*100
   
    # Get portfolio variables
    for name in (["business","retail","joint"]):
        this_period[f"{name}_change"] = this_period[name] - prev_period[name]

    # TODO look at negative saldo

    select = (this_period['portfoliototaal']>prev_period['portfoliototaal'])
    
    if (select_variables ==None):
        data = this_period.loc[select,'percdiff']
    
    else:
        data = this_period.loc[select,'percdiff']
    # add a variable for which type of data was added: dummy for business, joint,
    # retail
    
    # put in variables select: type of portfolios added, characteristics current, 
    # saldo of the individual types??
    
    #verschil van de logs pakken, met ook voorgaande saldo als verklarende 
    #variabele gebruiken (?)
    
    # Do a for loop outside this and add the thing to a df
    
    return data
    



# =============================================================================
# Methods for aggregating and transforming
# =============================================================================


def getlogs(Y):
    minY = Y.min()
    print(f"Taking log of {Y.name}. the minimum amount for this column is: {minY}")
    # add the smallest amount of Y
    logY = np.log(Y+1-minY)
    return logY


def aggregate_portfolio_types(df):
    """Aggregeer de activity variables en een aantal van de andere variabelen""" 
    
    
    # First get the sum of the number of products, logins, transacs PER business type
    for name in (["business","retail","joint"]):
        
        df[f'aantalproducten_totaal_{name}'] = df[[f"betalenyn_{name}",
                                    f"depositoyn_{name}",
                                    f"flexibelsparenyn_{name}",
                                    f"kwartaalsparenyn_{name}"]].sum(axis=1)
        
        df[f'logins_totaal_{name}'] = df[[f"aantalloginsapp_{name}",
                                    f"aantalloginsweb_{name}"]].sum(axis=1)
        
        df[f'aantaltransacties_totaal_{name}'] = df[[f"aantalbetaaltransacties_{name}",
                                    f"aantalatmtransacties_{name}",
                                    f"aantalpostransacties_{name}",
                                    f"aantalfueltransacties_{name}"]].sum(axis=1)
    
    # # For logins, activiteit, & transacties, we take the maximum
    # for variable in ['aantalloginsapp',
    #                  'aantalloginsweb',
    #                  'activitystatus',
    #                  'aantalbetaaltransacties',
    #                  'aantalatmtransacties',
    #                  'aantalpostransacties',
    #                  'aantalfueltransacties']:
    
    #     df[variable] = df[[f"{variable}_business",f"{variable}_retail", f"{variable}_joint"]].max(axis=1)
    
    ## Get sum of logins_totaal as well
    # df['logins_totaal'] = df[['aantalloginsapp','aantalloginsweb']].sum(axis=1) 
    
    ## Also get total transactions
    # df['aantaltransacties_totaal'] = df[['aantalbetaaltransacties',
    #                                      'aantalatmtransacties',
    #                                      'aantalpostransacties',
    #                                      'aantalfueltransacties']].sum(axis=1)
    
    #TODO note sum of max will always >= the max of sums - may not be fair?
    
    # Take the MAX of the logins, transactions and activity status 
    for var in ['logins_totaal','aantaltransacties_totaal','activitystatus']:
        df[f'{var}'] = df[[f"{var}_business",f"{var}_retail", 
                                f"{var}_joint"]].max(axis=1)

    # Max of the transactions
    # df['aantaltransacties_totaal'] = df[["aantaltransacties_totaal_business","aantaltransacties_totaal_retail", 
    #                         "aantaltransacties_totaal_joint"]].max(axis=1)
    
    # # max of the activity status 
    # df['activitystatus_totaal'] = df[["aantaltransacties_totaal_business","aantaltransacties_totaal_retail", 
    #                         "aantaltransacties_totaal_joint"]].max(axis=1)


    # Sum for the total account balance
    df['saldototaal'] = df[["saldototaal_business","saldototaal_retail", 
                            "saldototaal_joint"]].sum(axis=1)
  
    # Also get total number of products
    df['aantalproducten_totaal'] = df[["aantalproducten_totaal_business",
                                       "aantalproducten_totaal_retail", 
                                       "aantalproducten_totaal_joint"]].sum(axis=1)
    
    # Ook som van appyn meenemen, als moderating variable voor aantal logins?
    # -> moet dan alleen dat ook in de final dataset zetten, staat er nu niet nit
    
    return df
    


def transform_variables(df, separate_types = False):
    """Transform variables so that they are all around the same scale"""
    
    # Take the LOGARITHM of the account details, logins and transactions!
    for var in (["saldototaal","logins_totaal","aantaltransacties_totaal"]):
        if (separate_types):
            for name in (["business","retail","joint"]):                
                #df[f"log_{var}_{name}"] =  np.log(df[f"{var}_{name}"]+1)
                df[f"log_{var}_{name}"] = getlogs(df[f"{var}_{name}"])

        #df[f"log_{var}"] =  np.log(df[f"{var}"]+1) 
        df[f"log_{var}"] =  getlogs(df[f"{var}"]) 
        
    
    # Put a MAX on the business, portfolio, joint variables (3 per type max)
    for name in (["business","retail","joint"]):
        df[f"{name}_max"] = df[f"{name}"]

        df.loc[(df[f"{name}"]>3),f"{name}_max"] = "3"   # Should I make it a string or make it a number?
        # This only relevant for business and joint since retail does not have >1
        
    # TODO also put a max on the total number of products????
    
    # make geslacht into a dummy
    df["geslacht_dummy"] =1
    df.loc[(df["geslacht"]=="vrouw"),"geslacht_dummy"] = 0
    
    # make age into years (-> create categories?)
    today = date.today() 
    df["age"] = today.year - df["birthyear"]
    #df["agebins"] = pd.cut(df["age"], bins=6, labels=False)
    # can also define bins ourself: 
    bins = pd.IntervalIndex.from_tuples([ (0, 18), (18, 30), 
                                         (30, 45), (45, 60),
                                         (60, 75), (75, 200)])
    df["agebins"] = pd.cut(df["age"], bins, labels=False)
    
    
    