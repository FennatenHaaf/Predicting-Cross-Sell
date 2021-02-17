"""
Main method for the Knab Predicting Cross Sell case

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import knab_dataprocessor as KD
import utils
import additionalDataProcess as AD
import HMM_eff as ht

import pandas as pd
from os import path

if __name__ == "__main__":

# =============================================================================
# DEFINE PARAMETERS AND DIRECTORIES
# =============================================================================

    # Define where our input, output and intermediate data is stored
    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    
    save_intermediate_results = True # Save the intermediate outputs
    print_information = False # Print things like frequency tables or not

    quarterly = True # In which period do we want to aggregate the data?
    start_date = "2018-01-01" # From which moment onwards do we want to use
    # the information in the dataset
    end_date = None # Until which moment do we want to use the information
    subsample = True # Do we want to take a subsample 
    sample_size = 500 # The sample size
    finergy_segment = None # The finergy segment that we want to be in the sample
    # e.g.: "B04"
    
# =============================================================================
# DEFINE WHAT TO RUN
# =============================================================================
    
    cross_sec = False # Do we want to run the code for getting a single cross-sec
    time_series = False # Do we want to run the code for getting time series data
    saldo_data = True # Do we want to create the dataset for predicting saldo
    transform = False
    
    
# =============================================================================
# DEFINE SOME VARIABLE SETS TO USE FOR THE MODELS
# =============================================================================
    # Characteristics
    person_var = ['age_hh', # maybe don't need if we already have age
     'hh_size',
     'income',
     'educat4',
     'housetype',
     'finergy_tp',
     'lfase',
     'business',
     'huidigewaarde_klasse']
    
    person_transform = ["age",
                      "geslacht_dummy"]
    
    # perportfolio
    # aggregated portfolio things
    
    # activity
    activity_var = ["activitystatus",
                    "log_logins_totaal",
                    "log_aantaltransacties_totaal"]
    # also activity
    

# =============================================================================
# CREATE DATASETS
# =============================================================================
    
    #----------------INITIALISE DATASET CREATION-------------------
    start = utils.get_time()
    print(f"****Processing data, at {start}****")
    
    if (cross_sec | time_series): 
        #initialise dataprocessor
        processor = KD.dataProcessor(indirec, interdir, outdirec,
                                quarterly, start_date, end_date,
                                save_intermediate_results,
                                print_information
                                )
        # initialise the base linked data and the base Experian data which are
        # used to create the datasets
        processor.link_data() 
        #Create base experian information and select the ids used -> choose to
        # make subsample here! 
        processor.select_ids(subsample = subsample, sample_size = sample_size, 
                             finergy_segment = finergy_segment,
                             outname = "base_experian", filename = "valid_ids",
                             invalid = "invalid_ids",
                             use_file = True)
        
    #----------------MAKE CROSS-SECTIONAL DATASETS-------------------
    if cross_sec:
        # Make the base cross-section
        df_cross, cross_date = processor.create_base_cross_section(date_string="2020-12", 
                            next_period = False, outname = "cross_experian")
        # Now aggregate all of the information per portfolio
        df_cross_link = processor.create_cross_section_perportfolio(df_cross, cross_date, 
                                              outname = "df_cross_portfoliolink")
        # Aggregate all of the portfolio information per person ID
        df_out = processor.create_cross_section_perperson(df_cross, df_cross_link,
                                            cross_date, outname = "final_df_quarterly")
    
    #----------------MAKE TIME SERIES DATASETS-------------------
    if time_series:
        dflist = processor.time_series_from_cross(outname = "final_df")
        
    else:
        if (path.exists(f"{interdir}/final_df_2018Q1.csv")):
            print("****Reading df list of time series data from file****")
            
            dflist = [pd.read_csv(f"{interdir}/final_df_2018Q1.csv"),
                        pd.read_csv(f"{interdir}/final_df_2018Q2.csv"),
                        pd.read_csv(f"{interdir}/final_df_2018Q3.csv"),
                        pd.read_csv(f"{interdir}/final_df_2018Q4.csv"),
                        pd.read_csv(f"{interdir}/final_df_2019Q1.csv"),
                        pd.read_csv(f"{interdir}/final_df_2019Q2.csv"),
                        pd.read_csv(f"{interdir}/final_df_2019Q3.csv"),
                        pd.read_csv(f"{interdir}/final_df_2019Q4.csv"),
                        pd.read_csv(f"{interdir}/final_df_2020Q1.csv"),
                        pd.read_csv(f"{interdir}/final_df_2020Q2.csv"),
                        pd.read_csv(f"{interdir}/final_df_2020Q3.csv"),
                        pd.read_csv(f"{interdir}/final_df_2020Q4.csv")]
            
            print(f"sample size: {len(dflist[0])}")
            print(f"number of periods: {len(dflist)}")
           
            
    
    #--------------- GET DATA FOR REGRESSION ON SALDO ------------------
    if saldo_data:
        selection = ["percdiff"]
        diffdata = AD.create_saldo_data(dflist, interdir,
                                        filename= "saldopredict",
                                        select_variables = selection)

    #-----------------------------------------------------------------
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")
    
    
# =============================================================================
# RUN HMM MODEL
# =============================================================================
    
    #----------------AGGREGATE & TRANSFORM THE DATA-------------------
    if transform:    
       for i, df in enumerate(dflist):    
            dflist[i] = AD.aggregate_portfolio_types(dflist[i])
            dflist[i] = AD.transform_variables(dflist[i]) 

    # Hier code om het HMM model te runnen?
    # Definieer een lijst van characteristics om te gebruiken voor het
    # model!
    
    

    
    
