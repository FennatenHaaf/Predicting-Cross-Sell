"""
Main method for the Knab Predicting Cross Sell case

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import knab_dataprocessor as KD
import utils
import additionalDataProcess as AD
import HMM_eff as ht
import extra_functions_HMM_eff as ef

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
    subsample = False # Do we want to take a subsample
    sample_size = 500 # The sample size
    finergy_segment = None # The finergy segment that we want to be in the sample
    # e.g.: "B04"
    
# =============================================================================
# DEFINE WHAT TO RUN
# =============================================================================
    
    cross_sec = False # Do we want to run the code for getting a single cross-sec
    time_series = False # Do we want to run the code for getting time series data
    transform = True # Transform & aggregate the data
    saldo_data = True # Do we want to create the dataset for predicting saldo
    run_hmm = False
    
# =============================================================================
# DEFINE SOME VARIABLE SETS TO USE FOR THE MODELS
# =============================================================================
    
    #------ Portfolio types -------
    crosssell_types = [
        "business",
        "retail",
        "joint",
        "accountoverlay"
        ]
    
    crosssell_types_max = [ 
        # These are bounded (not larger than 3)
        "business_max",
        "retail_max",
        "joint_max",
        "accountoverlay_max"
        ]
    
    crosssell_types_max_nooverlay = [ 
        # These are bounded (not larger than 3)
        "business_max",
        "retail_max",
        "joint_max",
        ]

    #------ Account balances -------

    saldo_perportfolio = [
        "log_saldototaal_business",
        "log_saldototaal_retail", 
        "log_saldototaal_joint"
        ]
    
    saldo_total = [
        "log_saldototaal"
        ]
    
    # ------ Characteristics -------
    person_dummies = [
        # Experian variables:
        #'age_hh', # maybe don't need if we already have age
        #'hh_child',
        'hh_size',
        'income',
        'educat4',
        'housetype',
        #'finergy_tp',
        'lfase',
        #'business', don't use this, we dropped it
        'huidigewaarde_klasse',
        # Our own addition:
        "age_bins", 
        "geslacht"
        ]
        
    #------ Activity & transaction variables -------
    
    activity_dummies = [
        "activitystatus"
        ]
    
    activity_total = [
        "log_logins_totaal",
        "log_aantaltransacties_totaal",
        "aantalproducten_totaal"
        ]
    
    # activity variables per portfolio
    activity_perportfolio = []
    activity_dummies_perportfolio = []
    
    for var in (["business","retail","joint"]):  
       for name in activity_total:     
            activity_perportfolio.append(f"log_{var}_{name}")
       activity_dummies_perportfolio.append(f"activitystatus_{name}")
     
        
    #------ Business characteristics variables --------
    
    business_dummies = [
        "SBIname",
        "SBIsectorName",
        "businessAgeInYears_bins",
        "businessType",
        ]
    
    # TODO the dummies all  need to be tranformed later ! 
    # use: pd.get_dummies(df['col'], prefix=['col'])
    
    # also possible to get: logins web & logins app separately
    # & the transaction information separately
    
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
           
            
    #----------------AGGREGATE & TRANSFORM THE DATA-------------------
    if transform:    
       print(f"****Transforming datasets & adding some variables at {utils.get_time()}****")
       
       for i, df in enumerate(dflist):    
            df = dflist[i]
            df = AD.aggregate_portfolio_types(df)
            dflist[i]= AD.transform_variables(df) 
    
    
    #--------------- GET DATA FOR REGRESSION ON SALDO ------------------
    if (saldo_data & transform):
        print(f"****Create data for saldo prediction at {utils.get_time()}****")
        
        # Define which 'normal' variables to use in the dataset
        selection = activity_total # add activity vars
        
        # Define which dummy variables to use
        dummies = person_dummies # get experian characteristics
        dummies.extend(activity_dummies) # get activity status
        
        # Run the data creation
        predictdata = AD.create_saldo_data(dflist, interdir,
                                        filename= "saldopredict",
                                        select_variables = selection,
                                        dummy_variables = dummies)

    #-----------------------------------------------------------------
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")
    
    
# =============================================================================
# RUN HMM MODEL
# =============================================================================
    
    # Hier code om het HMM model te runnen?
    # Definieer een lijst van characteristics om te gebruiken voor het
    # model!
    if (run_hmm & transform):
        
        print(f"****Defining variables to use at {utils.get_time()}****")
        
        # Define the dependent variable
        name_dep_var_cross_sell = crosssell_types_max_nooverlay
        
        # Just need to get the dummies for activity status for now
        for i, df in enumerate(dflist):            
            df = dflist[i]
            dummies, dummynames =  AD.make_dummies(df,
                                                 activity_dummies,
                                                 drop_first = True)
            df[dummynames] = dummies[dummynames]
        
        activity_variables = dummynames #activity status 1.0 is de base case
        activity_variables.extend(activity_total)
        
        # Say which covariates we are going to use
        name_covariates = activity_variables
        
        # take a subset of the number of periods, just to test
        df_periods  = dflist[:3] # only use 3 periods
        
        #Define number of segments
        n_segments = 4
        
        #---------------- RUN THE HMM MODEL ---------------
        
        startmodel = utils.get_time()
        #print(f"****Running HMM at {startmodel}****")
        
        test_cross_sell = ht.HMM_eff(df_periods, name_dep_var_cross_sell, 
                                     name_covariates, covariates = True)

        
        param_cross, alpha_cross, shapes_cross = test_cross_sell.EM(n_segments, 
                                                                    max_method = 'Nelder-Mead')
       
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(test_cross_sell, 
                                                                          n_segments, 
                                                                          param_cross, 
                                                                          shapes_cross)
        
        p_js = ef.prob_p_js(test_cross_sell, param_cross, shapes_cross, n_segments)
    
        endmodel = utils.get_time()
        diff = utils.get_time_diff(startmodel,endmodel)
        print(f"HMM finished! Total time: {diff}")

    
