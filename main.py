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
import dataInsight as DI
import predictsaldo as ps

from scipy.stats.distributions import chi2
import pandas as pd
import numpy as np
from os import path

if __name__ == "__main__":

# =============================================================================
# DEFINE PARAMETERS AND DIRECTORIES
# =============================================================================

    # Define where our input, output and intermediate data is stored
    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"

    save_intermediate_results = False # Save the intermediate outputs
    print_information = False # Print things like frequency tables or not

    quarterly = True # In which period do we want to aggregate the data?
    #start_date = "2020-01-01" # From which moment onwards do we want to use
    start_date = "2018-01-01" # From which moment onwards do we want to use

    # the information in the dataset
    end_date = None # Until which moment do we want to use the information
    #end_date = "2020-12-31" # Until which moment do we want to use the information
    subsample = False # Do we want to take a subsample
    sample_size = 1000 # The sample size
    finergy_segment = "B04"
    #"B04" # The finergy segment that we want to be in the sample
    # e.g.: "B04" - otherwise make it None
    
    # How to save the final result
    if (finergy_segment != None):
        if subsample:
            final_name = f"final_df_fin{finergy_segment}_n{sample_size}"
            #saldo_name =  f"saldopredict_fin{finergy_segment}_n{sample_size}"
        else:
            final_name = f"final_df_fin{finergy_segment}"
           #saldo_name = f"saldopredict_fin{finergy_segment}"
    else:
        if subsample:
            final_name = f"final_df_n{sample_size}"
            #saldo_name = f"saldopredict_n{sample_size}"
        else:
            final_name = "final_df"
           # saldo_name = f"saldopredict"
            
# =============================================================================
# DEFINE WHAT TO RUN
# =============================================================================
    
    time_series = False # Do we want to run the code for getting time series data
    visualize_data = False # make some graphs and figures
    
    run_hmm = False
    run_cross_sell = True # do we want to run the model for cross sell or activity
    interpret = True #Do we want to interpret variables
    saldopredict = True # Do we want to run the methods for predicting saldo

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
    
    crosssell_types_dummies = [
        "business_dummy",
        "retail_dummy",
        "joint_dummy",
        "accountoverlay_dummy"
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
    saldo_total_bin = [
        "saldototaal_bins"
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
        #"aantalproducten_totaal"
        ]
    activity_total_dummies = [
        "log_logins_totaal_bins",
        "log_aantaltransacties_totaal_bins",
        #"aantalproducten_totaal_bins"
        ]
    
    
# =============================================================================
# CREATE DATASETS
# =============================================================================
    
    #----------------INITIALISE DATASET CREATION-------------------
    start = utils.get_time()
    print(f"****Processing data, at {start}****")
    
    if (time_series): 
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
        
    
    #----------------MAKE TIME SERIES DATASETS-------------------
    #if time_series:
        print(f"final name:{final_name}")
        dflist = processor.time_series_from_cross(outname = final_name)
        
    else:
        name = "final_df_finB04" # adjust this name to specify which files to read
        #name = "final_df_n500"
        if (path.exists(f"{interdir}/{name}_2018Q1.csv")):
            print("****Reading df list of time series data from file****")
            
            dflist = [pd.read_csv(f"{interdir}/{name}_2018Q1.csv"),
                        pd.read_csv(f"{interdir}/{name}_2018Q2.csv"),
                        pd.read_csv(f"{interdir}/{name}_2018Q3.csv"),
                        pd.read_csv(f"{interdir}/{name}_2018Q4.csv"),
                        pd.read_csv(f"{interdir}/{name}_2019Q1.csv"),
                        pd.read_csv(f"{interdir}/{name}_2019Q2.csv"),
                        pd.read_csv(f"{interdir}/{name}_2019Q3.csv"),
                        pd.read_csv(f"{interdir}/{name}_2019Q4.csv"),
                        pd.read_csv(f"{interdir}/{name}_2020Q1.csv"),
                        pd.read_csv(f"{interdir}/{name}_2020Q2.csv"),
                        pd.read_csv(f"{interdir}/{name}_2020Q3.csv"),
                        pd.read_csv(f"{interdir}/{name}_2020Q4.csv")]
            
            print(f"sample size: {len(dflist[0])}")
            print(f"number of periods: {len(dflist)}")
           
            
    #----------------AGGREGATE & TRANSFORM THE DATA-------------------
    additdata = AD.AdditionalDataProcess(indirec,interdir,outdirec)
 
    print(f"****Transforming datasets & adding some variables at {utils.get_time()}****")
    
    for i, df in enumerate(dflist):    
         df = dflist[i]
         df = additdata.aggregate_portfolio_types(df)
         dflist[i]= additdata.transform_variables(df)
 
     
    print("Doing a check that the ID columns are the same, and getting minimum saldo")
    
    for i, df in enumerate(dflist):    
        if (i==0): 
            dfold = dflist[i]
        else:
            dfnew = dflist[i]
            if (dfold["personid"].equals(dfnew["personid"])):
                #print("check") 
                check=1 # we do nothing
            else:
                print("noooooooooooo")
            dfold = dfnew
              
    #-----------------------------------------------------------------
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")
    
    
# =============================================================================
# VISUALIZE DATA
# =============================================================================

    if visualize_data:
        print("*****Visualising data*****")
        df = dflist[11] # use the last time period
        visualize_variables = ["age_bins", "geslacht",
                               "hh_size",
                               "income",
                               "business_max",
                               "retail_max",
                               "joint_max",
                               "accountoverlay_max",
                               "activitystatus",
                               "saldototaal_bins",
                               "SBIsectorName",
                               "businessAgeInYears_bins",
                               "businessType"]   
        for var in visualize_variables:
            print(f"visualising {var}")
            DI.plotCategorical(df, var)
            counts = df[var].value_counts().rename_axis(var).to_frame("total count").reset_index(level=0)
            print(counts)
        DI.plotCoocc(df, crosssell_types_dummies)
        
        print("*****Visualising cross-sell difference data*****")
        # Now visualise cross sell events from the full dataset
       
        diffdata = additdata.create_crosssell_data(dflist, interdir) 
        visualize_variables = ["business_change_dummy", "retail_change_dummy",
                               "joint_change_dummy",
                               "accountoverlay_change_dummy",
                              ]
        # ["business_change",
        #                        "retail_change",
        #                        "joint_change",
        #                        "accountoverlay_change",
        #                       ]   
        
        for var in visualize_variables:
            print(f"visualising {var}")
            DI.plotCategorical(diffdata, var)
            counts = df[var].value_counts().rename_axis(var).to_frame("total count").reset_index(level=0)
            print(counts)
            
        #DI.plotCoocc(df,  visualize_variables  )
        #TODO er gaat nog iets mis hier!
        
    
# =============================================================================
# RUN HMM MODEL
# =============================================================================
    
    #---------MAKE PERSONAL VARIABLES (ONLY INCOME, AGE, GENDER) ---------
    # we don't use all experian variables yet
    dummies_personal = ["income","age_bins","geslacht"] 
    for i, df in enumerate(dflist):   
        dummies, dummynames =  additdata.make_dummies(df,
                                             dummies_personal,
                                             drop_first = False)
        df[dummynames] = dummies[dummynames]
    print("Dummy variables made:")
    print(dummynames)
    # get the dummy names without base cases        
    base_cases = ['income_1.0','age_bins_(0, 18]','age_bins_(18, 30]',
                  'geslacht_Man','geslacht_Man(nen) en vrouw(en)']

    personal_variables = [e for e in dummynames if e not in base_cases]

    #--------------------GET FULL SET OF COVARIATES----------------------

    # First add the continuous variables
    full_covariates = ["log_logins_totaal","log_aantaltransacties_totaal"] 

    # Time to process dummies again
    dummies_personal = ["income", "age_bins", "geslacht", "hh_size",
                        "saldototaal_bins", "businessType"] 
    for i, df in enumerate(dflist):   
        dummies, dummynames =  additdata.make_dummies(df,
                                             dummies_personal,
                                             drop_first = False)
        df[dummynames] = dummies[dummynames]
    print("Dummy variables made:")
    print(dummynames)
    # get the dummy names without base cases        
    base_cases = ['income_1.0','age_bins_(18, 30]','age_bins_(0, 18]',
                  'geslacht_Man','geslacht_Man(nen) en vrouw(en)',
                  'hh_size_1.0','saldototaal_bins_(0.0, 100.0]', 
                  ]
    dummies_final = [e for e in dummynames if e not in base_cases]
    full_covariates.extend(dummies_final)



    if (run_hmm):
        #---------------- SELECT VARIABLES ---------------
        print(f"****Defining variables to use at {utils.get_time()}****")

        if (run_cross_sell): 
            print("Running cross sell model")
            # Define the dependent variable
            name_dep_var = crosssell_types_max
            # Say which covariates we are going to use
            name_covariates = full_covariates
            # take a subset of the number of periods
            df_periods  = dflist[4:10]  # use periods 5,6,7,8,9 and 10
            #Define number of segments
            n_segments = 6
            reg = 0.05 # Regularization term
            max_method = 'Nelder-Mead'
            
            outname = f"crosssell_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
            
            
        else: # run the activity model!
            print("Running activity model")
            # Define the dependent variable
            activity_dummies.extend(activity_total_dummies)
            name_dep_var = activity_dummies # activitystatus, logins, transactions
            # Say which covariates we are going to use
            name_covariates = personal_variables # income, age, gender
            # take a subset of the number of periods 
            df_periods  = dflist[4:10] # use periods 5,6,7,8,9 and 10
            #Define number of segments
            n_segments = 3
            reg = 0.1 # Regularization term
            max_method = 'Nelder-Mead'
            
            outname = f"activity_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
        
        
        #---------------- Enter initial parameters ---------------
        
        initial_param = None
        param_out = np.array(
            [4.79308382e+00, 1.92205555e+00, -7.58345319e-02, 5.20282321e-09, -9.28609227e-02, -3.64672986e-05, -2.83188344e-01,
             -3.14202992e-05, -1.40888469e-04, 4.59771090e-01, 7.91803874e-01, -1.10168101e-11, -2.07837227e-06, -8.97969905e-13,
             -1.92018221e-05, 8.97473776e-09, 4.60067132e-08, -2.60988672e-01, -4.05245774e-02, 4.09563748e-14, -4.30608984e-08,
             3.16326803e-09, -3.32499529e-06, 5.38245319e-10, 1.55317030e-08, 9.33315450e-04, 4.60073358e-09, -1.91757814e-21,
             2.59544048e+00, -3.37017665e-01, -6.33985812e-07, 8.89790557e-01, 1.00606593e-01, 7.32529787e-02, -6.90073448e-09,
             2.91859002e-01, -3.49695012e-05, 4.56423682e-13, 1.60656229e-01, 1.62580728e+00, 1.36404933e-06, -6.67788094e-01,
             3.82779735e-04, -7.87433480e-06, 3.12918780e+00, 2.10031821e+00, 6.83428734e-01, -1.05785791e-01, -6.38737267e-07,
             5.60993429e-03, 3.34361499e-08, 8.84321765e-08, 3.80484772e-03, -2.69393003e-02, -5.82298825e-06, 2.89326823e+00,
             3.75046919e-06, -1.21131746e-09, -2.34827033e-04, 7.33588967e-08, 2.07683471e-12, 2.69146198e-06, -1.78432662e-12,
             4.40376958e-03, -3.33462538e-10, -4.79199842e-10, 8.95936427e-01, 1.01219991e-11, -1.14202711e-06, 7.38231179e-06,
             5.70741330e-05, 1.79099917e-01, -1.98831541e-14, 2.04031909e-09, 6.88345365e-08, 1.05187144e-10, 4.72929013e-07,
             3.43863612e-06, -5.00531845e-05, 5.89205611e-03, -1.20917365e-06, 2.51067634e-07, -7.09150559e-07, 1.73044924e-06,
             -5.86778236e-09, 6.56622040e-10, 5.29638046e-06, -1.96454345e-02, 1.44747767e-06, -1.09535234e-14, 7.97904463e-06,
             -1.18833904e-05, 9.27114585e-06, -2.18698076e-11, -2.37879436e-06, 6.45353741e-14, -7.58778925e-09, 5.81195778e-11,
             5.25954417e-09, -1.27915717e-09, -7.43012671e-08, -2.30462717e-09, -6.89347485e-05, 2.70942062e-05, -1.08238611e-05,
             1.81939637e-03, 1.30233522e-14, 1.87977759e-05, -1.19185429e-08, 1.35430010e+00, -1.32567127e-03, 9.98752707e-06,
             2.37445345e-03, -2.89005078e-09, 5.94198990e-05, -1.00172345e+00, 1.83587480e-05, 4.06100423e-05, 5.46084808e-09,
             7.31120326e-07, -4.56841307e-10, -4.06845761e-08, -1.27497666e-05, -4.25646703e-09, -4.98828585e-10, -1.67391383e-15,
             2.42258651e-04, -7.35266297e-08, -2.29463292e-10, -7.84841508e-04, 1.69499275e+01, 1.77169969e+01, 1.32708326e+01,
             1.58147852e+01, 1.63061859e+01, 1.34137368e-06, 5.91693685e+00, -9.62369217e+00, 4.26639182e+00, -2.67632996e-08,
             9.89953763e-08, 4.91640657e+00, 1.17558628e-01, -1.02324162e+00, -3.77166373e-04, 3.31508457e+01, 3.80102613e-09,
             5.92990069e-08, 7.97025733e-01, 3.96517682e-11, -3.12245043e+00, 2.83527752e+01, -2.39007085e-17, -5.52830841e-07,
             -4.13057372e+00, -2.74026682e+01, -7.37916364e+00, -2.26518368e-10, 2.25181996e-07, 1.07535214e-06, 2.03308794e+00,
             -5.48654766e+00, 2.61110951e-03, 2.73558196e+01, -4.81907612e-07, -7.28441153e-04, 9.56122554e-01, 3.34043900e+00,
             1.42928540e-06, -2.01746927e-05, -2.68664177e+00, 9.90916269e-01, 5.78159808e+00, 3.61843220e+00, 6.33895231e-05,
             1.30463992e+00, -3.19808143e+00, -6.31021805e-08, 5.17775935e-01, -8.38174134e-01, 6.50910589e+00, -1.09517254e-08,
             4.42065570e+00, 3.74835071e+00, 2.94868459e+00, 1.65487774e+00, 1.18256887e-08, -1.91223478e+01, 2.02730635e-09,
             -2.20010098e-04, -9.14329169e-09, 1.08540837e+00, 4.36100767e-03, -5.97974512e-06, -4.70994590e-01, 1.06476850e-07,
             1.30943209e-08, 9.59343481e-01, -2.44329337e-07, -1.33666255e-03, 1.19757132e-07, -6.92187647e-01, -3.13330080e+00,
             -6.09433734e-01, -4.55281196e-01, 1.23043578e-01, 1.40819095e+00, -3.02339426e-11, 4.57784479e-14, 9.55770654e-01,
             -1.69262365e-07, 1.43218585e-05, -1.77998661e-05, -1.44491029e+01, -2.69164305e-08, -6.49302645e-07, 7.89749917e-07,
             9.95180879e-01, 5.65515504e-01, 3.93598582e-01, 2.50380530e-01, -1.69759340e-13, 1.78040378e+00, 1.00631849e+00,
             1.31222067e+00, -4.15539237e-12, -1.98374836e-02, -3.15338938e-01, -3.37548541e-03, 1.29077001e-06, -9.08158612e-10,
             2.64129345e-01, 1.17599265e-04, -1.44106945e+00, 2.91785626e-09, 6.33505853e-01, -1.42507673e-16, 1.69009742e-01,
             1.20580158e-09, -1.46619969e+01, -1.76279671e-03, 7.71745094e-05, 3.49609189e-07, -1.99672772e-09, 7.13849768e-01,
             -4.39298516e-09, 1.18163400e-09, 1.02038467e+01, 1.25890200e+01, 1.05119579e+00, 1.42648509e-02, -2.03193934e+00,
             -6.79950304e-10, -5.99352290e-06, 1.16682127e+01, -3.86739237e-07, 1.50358048e-10, 5.79545830e+00, 9.95120290e-13,
             -1.51330543e-10, 3.12895720e-01, -1.31413184e-02, -8.69355750e-13, 1.01865546e+00, -3.42227715e-01, 6.80728241e-16,
             1.47175912e-05, 1.56142568e-04, 5.09916881e-10, 1.49652449e+00, 1.06346523e-07, 1.99758765e-04, -5.84687454e-06,
             2.32835609e-04, 1.67904040e-04, -1.29118892e-02, 1.56751665e-06, 9.37005673e-08, 1.14213049e-07, -1.31038147e-05,
             -8.19497067e-05, -1.70191205e-06, 4.91787421e-06, 1.99059885e+00, -3.45056772e-07, -7.81862708e-11, 2.06642525e-08,
             2.11455990e-04, -1.57208908e+00, 4.00594995e-01, 1.49087616e+01, 1.57837757e-09, 1.24307122e+01, 1.37391530e+01,
             1.30282360e+01, -2.96141768e+00, 1.93781246e-02, 1.37049358e-07, 8.52188135e+00, -8.90277591e+00, -1.43469342e-04,
             -1.02598171e+01, -1.56867632e+00, -2.53890368e-05, -2.33637812e-02, 1.63978850e-11, -1.29639243e-07, 6.74980933e-04,
             -8.37845901e+00, -3.26955969e+00, 8.72804634e-13, -1.14941195e-02, 1.88944809e-13, 1.08617400e+01, -1.54763623e-04,
             -1.03923744e+01, 1.31718021e-07, -1.00072085e-09, -2.54910957e-08, 2.11024738e-11, -9.06694548e+00, -1.53336815e-04,
             -1.02068664e+01, 1.54348354e+01, 1.64659481e+01, 1.61047625e+01, -7.61593325e-01, -4.85654688e-11, 8.36451741e+00,
             -2.75606391e+00, -7.20104389e+00, -9.73589387e+00, -1.03949371e+01, 2.22968943e+00, -1.09602627e+00, -9.23933520e+00,
             1.17494542e-16])

        #---------------- RUN THE HMM MODEL ---------------  
        
        startmodel = utils.get_time()
        print(f"****Running HMM at {startmodel}****")
        print(f"dependent variable: {name_dep_var}")
        print(f"covariates: {name_covariates}")
        print(f"number of periods: {len(df_periods)}")
        print(f"number of segments: {n_segments}")
        print(f"number of people: {len(dflist[0])}")
        print(f"regularization term: {reg}")
        
        
        # Note: the input datasets have to be sorted / have equal ID columns!
        hmm = ht.HMM_eff(outdirec, outname,
                         df_periods, reg, max_method,
                         name_dep_var, name_covariates, 
                         covariates = True,
                         iterprint = True,
                         initparam = initial_param)

        # Run the EM algorithm - max method can be Nelder-Mead or BFGS
        param_cross,alpha_cross,beta_cross,shapes_cross,hess_inv = hmm.EM(n_segments, 
                                                             max_method = max_method,
                                                             reg_term = reg,
                                                             random_starting_points = True)  

        # Transform the output back to the specific parameter matrices
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(hmm, 
                                                                          n_segments, 
                                                                          param_cross, 
                                                                          shapes_cross)
    
        endmodel = utils.get_time()
        diff = utils.get_time_diff(startmodel,endmodel)
        print(f"HMM finished! Total time: {diff}")



# =============================================================================
# INTERPRET PARAMETERS
# =============================================================================

    if (interpret):
        print("****Checking HMM output****")
        
        if (not run_hmm): # read in parameters if we have not run hmm
            print("-----Reading in existing parameters-----")
             
            source = "finalActivity"
            #source = "crosssell4seg"
            #source = "crosssell5seg"        
            
            if (source == "finalActivity"):
                
                # note: dataset used is "final_df_finB04"
                parameters_in = np.array(
                    [3.45964482e-05, -5.95053017e-04, -1.01707154e+00, -8.89764111e-04, -7.64125808e-01, 7.52693657e-05,
                     -2.04091137e-16, -5.13153589e+00, 1.46161913e+00, 4.61799705e-04, -3.06432797e-04, -1.01685186e-04,
                     1.56473814e-05, -3.23723949e+00, 3.93821215e-01, -3.36437535e-05, -3.08204756e-04, 3.08238111e-06, 4.43163053e-08,
                     3.22788171e+00, 1.47269763e+00, 1.15266559e-04, -1.29857217e-06, 2.89076870e+01, 2.52897017e-05, 2.45215985e+01,
                     2.46398987e-08, 5.43649329e+00, -1.26835397e+00, 3.05540346e-01, -1.63492742e-04, 4.40136704e+00, -5.14369363e-04,
                     2.10332941e+00, -3.67557972e+00, 1.32053694e+00, -1.94006180e+00, -2.58182273e-05, -1.64448312e+00,
                     -9.72071725e-07, -4.47298442e-01, 2.81418175e+00, -4.27496510e-04, -3.90112560e-04, -2.46829325e+00,
                     -4.50948360e-03, 5.84382348e+00, 2.16838703e+00, 1.87682395e-01, 1.77765842e+01, 8.94753990e-09, 2.33722077e-10,
                     9.57805122e+00, -6.55235990e-01, 1.55045700e+01, -1.32778125e+00, 2.02596439e-01, -1.02892638e-05, 9.20574886e-01,
                     -1.82353830e-01, 3.66289597e-04, -1.70343897e+00, 2.42745313e+00, -1.67392557e-04, 7.81450078e-04, 3.11011873e-06,
                     2.86964770e-04, -7.82324151e-01, 1.99115541e+00, -2.37238747e-04, 4.10889870e-05, -5.72096374e-04,
                     -5.17937054e-05, 4.71548164e+00, 1.61665340e+00, -3.90264300e-01, -1.19498654e-03, -3.02782244e-07,
                     -1.26749320e+01, 1.48529585e-15, -1.65057568e+01, 8.54597218e-01, -1.97499167e-06, 1.51620975e-04,
                     -5.25331304e-04, 1.44526716e-04, 5.63062247e-04, -4.75204952e-04, -1.54755035e-05, 9.74492205e-01, 1.68915524e+00,
                     8.30147216e+00, 6.23140699e-04, 7.63199795e-05, 9.45925241e-01, -2.97332985e-06, 3.56657235e-06, -1.60074574e-07,
                     -1.97283503e-05, 7.16401937e-05, 2.74850783e-05, -7.83424173e+00, -1.07287186e-04, -4.02059419e-07,
                     -1.85657463e-04, -3.94185351e-01, -3.42039721e-06, 4.25803177e-05, 5.16238280e-05, -1.40402195e-05,
                     -4.46718150e-06, 1.83502986e-04, 8.40511118e-05, -3.30517186e-04, 9.00965817e-04, -6.29315032e-05, 5.24131548e+00,
                     1.71677140e-06, -9.22600786e-04, 9.12416408e-01, 3.71321354e-01, 2.19118002e-04, -3.57747915e-04, -1.62358636e-05,
                     2.02460508e-01, 3.39886361e-04, 5.12208599e-04, -3.00347603e-07, 1.27998892e-04, -1.93466775e-01, -8.17274196e-13,
                     1.11033400e-06, -2.75951301e-06, -1.67568433e-07, -7.30824895e-07, 1.07386073e-05, -2.60818334e-05,
                     4.27063683e-06, 2.81090789e-03, 5.84412654e-04, 1.62310672e+01, 1.74531679e+01, -3.10924519e-05, 1.34364160e+01,
                     1.32697252e+01, 1.19382198e+00, 3.76931030e-01, -1.10272273e-03, 1.58061171e+00, 1.36204867e+00, 1.88374818e+00,
                     5.21299490e-01, -3.95856531e-10, -1.51015778e-05, 1.76684917e-07, -3.28265940e-08, 3.16214611e-04, 1.23009849e-04,
                     1.88294418e-01, 1.49218317e-04, -2.63911731e-04, -1.28471865e-04, -5.52817671e-04, 1.65933514e-04, 8.73216476e-04,
                     -2.28018703e-04, -7.13803564e-03, 8.30829344e-09, -9.29670194e+00, 8.65765145e-14, 4.10642709e-04, 8.70954936e+00,
                     7.24865245e+00, -6.45411152e+00, 7.13514435e-02, -6.65518311e-01, 8.06614773e-01, -6.39196817e+00, 9.62442928e+00,
                     1.14488889e+01, 7.72161856e-01, 1.72495903e-01, -7.70915449e-01, 9.71104363e-01, -5.97076228e-01, -1.15817066e+01,
                     -8.37944596e+00, -1.79967357e+01, 1.39282306e+00, -4.64875777e-01, -9.71724875e+00, -9.50947676e+00,
                     2.29431767e-02, -3.16229218e+00, 6.45131773e+00, 1.30575579e+00, -2.40785912e-01, -1.82436247e+00,
                     -2.03783144e+00])

                # Define the dependent variable
                activity_dummies.extend(activity_total_dummies)
                name_dep_var = activity_dummies # activitystatus, logins, transactions
                # Say which covariates we are going to use
                name_covariates = personal_variables # income, age, gender
                # take a subset of the number of periods 
                df_periods  = dflist[4:10] # use periods 5,6,7,8,9 and 10
                #Define number of segments
                n_segments = 6
                reg = 0.1 # Regularization term
                max_method = 'Nelder-Mead'
                run_cross_sell = False
                
            if (source == "crosssell4seg"):
                
                # ivan results met loglikelihood 11599.8611774401
                # note: dataset used is "final_df_finB04"
                param_cross = np.array([ 3.45964482e-05, -5.95053017e-04, -1.01707154e+00, -8.89764111e-04, -7.64125808e-01, 7.52693657e-05, -2.04091137e-16, -5.13153589e+00, 1.46161913e+00, 4.61799705e-04, -3.06432797e-04, -1.01685186e-04, 1.56473814e-05, -3.23723949e+00, 3.93821215e-01, -3.36437535e-05, -3.08204756e-04, 3.08238111e-06, 4.43163053e-08, 3.22788171e+00, 1.47269763e+00, 1.15266559e-04, -1.29857217e-06, 2.89076870e+01, 2.52897017e-05, 2.45215985e+01, 2.46398987e-08, 5.43649329e+00, -1.26835397e+00, 3.05540346e-01, -1.63492742e-04, 4.40136704e+00, -5.14369363e-04,
                                        2.10332941e+00, -3.67557972e+00, 1.32053694e+00, -1.94006180e+00, -2.58182273e-05, -1.64448312e+00, -9.72071725e-07, -4.47298442e-01, 2.81418175e+00, -4.27496510e-04, -3.90112560e-04, -2.46829325e+00, -4.50948360e-03, 5.84382348e+00, 2.16838703e+00, 1.87682395e-01, 1.77765842e+01, 8.94753990e-09, 2.33722077e-10, 9.57805122e+00, -6.55235990e-01, 1.55045700e+01, -1.32778125e+00, 2.02596439e-01, -1.02892638e-05, 9.20574886e-01, -1.82353830e-01, 3.66289597e-04, -1.70343897e+00, 2.42745313e+00, -1.67392557e-04, 7.81450078e-04, 3.11011873e-06, 
                                        2.86964770e-04, -7.82324151e-01, 1.99115541e+00, -2.37238747e-04, 4.10889870e-05, -5.72096374e-04, -5.17937054e-05, 4.71548164e+00, 1.61665340e+00, -3.90264300e-01, -1.19498654e-03, -3.02782244e-07, -1.26749320e+01, 1.48529585e-15, -1.65057568e+01, 8.54597218e-01, -1.97499167e-06, 1.51620975e-04, -5.25331304e-04, 1.44526716e-04, 5.63062247e-04, -4.75204952e-04, -1.54755035e-05, 9.74492205e-01, 1.68915524e+00, 8.30147216e+00, 6.23140699e-04, 7.63199795e-05, 9.45925241e-01, -2.97332985e-06, 3.56657235e-06, -1.60074574e-07, -1.97283503e-05, 
                                        7.16401937e-05, 2.74850783e-05, -7.83424173e+00, -1.07287186e-04, -4.02059419e-07, -1.85657463e-04, -3.94185351e-01, -3.42039721e-06, 4.25803177e-05, 5.16238280e-05, -1.40402195e-05, -4.46718150e-06, 1.83502986e-04, 8.40511118e-05, -3.30517186e-04, 9.00965817e-04, -6.29315032e-05, 5.24131548e+00, 1.71677140e-06, -9.22600786e-04, 9.12416408e-01, 3.71321354e-01, 2.19118002e-04, -3.57747915e-04, -1.62358636e-05, 2.02460508e-01, 3.39886361e-04, 5.12208599e-04, -3.00347603e-07, 1.27998892e-04, -1.93466775e-01, -8.17274196e-13, 1.11033400e-06, 
                                        -2.75951301e-06, -1.67568433e-07, -7.30824895e-07, 1.07386073e-05, -2.60818334e-05, 4.27063683e-06, 2.81090789e-03, 5.84412654e-04, 1.62310672e+01, 1.74531679e+01, -3.10924519e-05, 1.34364160e+01, 1.32697252e+01, 1.19382198e+00, 3.76931030e-01, -1.10272273e-03, 1.58061171e+00, 1.36204867e+00, 1.88374818e+00, 5.21299490e-01, -3.95856531e-10, -1.51015778e-05, 1.76684917e-07, -3.28265940e-08, 3.16214611e-04, 1.23009849e-04, 1.88294418e-01, 1.49218317e-04, -2.63911731e-04, -1.28471865e-04, -5.52817671e-04, 1.65933514e-04, 8.73216476e-04, 
                                        -2.28018703e-04, -7.13803564e-03, 8.30829344e-09, -9.29670194e+00, 8.65765145e-14, 4.10642709e-04, 8.70954936e+00, 7.24865245e+00, -6.45411152e+00, 7.13514435e-02, -6.65518311e-01, 8.06614773e-01, -6.39196817e+00, 9.62442928e+00, 1.14488889e+01, 7.72161856e-01, 1.72495903e-01, -7.70915449e-01, 9.71104363e-01, -5.97076228e-01, -1.15817066e+01, -8.37944596e+00, -1.79967357e+01, 1.39282306e+00, -4.64875777e-01, -9.71724875e+00, -9.50947676e+00, 2.29431767e-02, -3.16229218e+00, 6.45131773e+00, 1.30575579e+00, -2.40785912e-01, -1.82436247e+00,
                                        -2.03783144e+00]) 

                # Define the dependent variable
                name_dep_var = crosssell_types_max
                # Say which covariates we are going to use
                name_covariates = full_covariates
                # take a subset of the number of periods
                df_periods  = dflist[4:10]  # use periods 5,6,7,8,9 and 10
                #Define number of segments
                n_segments = 4
                reg = 0.05 # Regularization term
                max_method = 'Nelder-Mead'
                run_cross_sell = True
                
                
            if (source == "crosssell5seg"):
                # fenna results met loglikelihood 8502.154086118475
                # note: dataset used is "final_df_finB04"
                param_cross = np.array([ 4.19867310e+00, 6.06015209e+00, 2.37976143e-01, -6.64865721e-01, -4.12053659e-05, -6.80088543e-02, -4.11376847e-03, 3.88857655e-01, 2.00450931e-05, 2.70608141e-07, -5.61241559e-01, 8.43083490e-05, 1.40596669e-06, -1.07061592e-08, 8.04423755e-02, 6.13768356e-02, -1.28354986e-04, -1.32369389e+00, -6.80543286e-07, 3.80317376e-03, -3.52459625e-02, -2.65383337e-02, -1.26437996e+01, -9.70072510e+00, 3.74489452e-05, -1.84209430e-08, -1.21529891e-10, -1.63393508e+00, 6.31001746e+00, 1.85973035e-01, -4.60900509e-01, 7.23505477e-01,
                                        -5.79910905e-11, -2.78709427e-05, -1.92956881e+00, -2.01952351e+00, -3.50593768e+00, -3.30149668e+00, 8.02344788e-07, 1.37244256e+00, -4.54699358e-15, 7.18617469e-01, -1.16574293e+00, -4.76513301e-05, -4.58481882e+00, -9.43100317e-01, 3.30139559e-01, -1.46519473e+00, 3.07170955e-07, 3.51960406e+00, 1.88931152e+00, 1.28357709e+01, 1.59580860e+01, 1.65564415e+01, 7.96072920e+00, 5.36006759e+00, 1.18622758e-02, 3.59979256e-05, 5.13486582e-01, 2.83701522e-01, 1.01562202e-01, -4.17664495e-02, 1.54737505e-01, 2.04681547e-01,
                                        -5.36179216e-17, 4.17843227e-01, -3.68866810e-01, -1.51643262e-01, -4.71683549e-03, -2.44488990e-04, -1.07947955e-02, -4.21503291e-01, 6.60551326e-01, 3.37311599e-01, 2.19870127e-01, 3.30004375e-01, -1.28057777e+01, -8.80375263e+00, 3.47121964e-05, 2.68272331e-05, 6.21237363e-06, -3.59373459e-01, 4.68925560e+00, -1.01379959e+00, -2.90753957e-06, -2.02851290e+00, 2.37750017e-03, -3.47973644e-01, 2.27099315e-01, 1.83661799e+00, 3.92795598e+00, 3.00608272e+00, -1.90014319e-03, -4.83057482e+00, 2.12331386e-13, -2.36358215e+00,
                                        1.70372776e-06, 1.38746349e-03, -1.70887363e-08, 2.14184329e-01, -2.81959270e-03, 7.35118180e-01, -1.13808522e-06, 7.05625839e+00, 1.92285717e+01, -5.41823924e-05, -2.50050767e-05, -2.46973589e-07, 4.80097531e+00, -3.31949076e+00, -6.49927513e-04, -5.44693344e+00, -4.97340890e-03, 7.01561306e-07, -1.51167574e+00, 5.27260920e-04, -4.21267979e+00, 1.67109460e-03, 
                                        5.15070350e+00, 2.47285507e+00, 1.01765190e+01, 1.48357221e-07, 3.15744426e-03, 6.95551206e-06, 4.41770872e-09, -3.76411655e-08, 1.44150438e-12, -1.68356833e-04, 6.20942979e+00, -2.54743875e-04, 6.20261687e-01, 2.96150320e+00, 1.84176795e+00, 2.56780672e+00, 1.12594573e+00, 8.83447047e-01, 1.14236507e+00, 2.47078658e-07, -4.45529788e-03, -4.47564065e-01, 3.70626452e-08, 2.37388164e-04, -3.90460792e-01, 4.09616521e-05, -1.18740436e+00, -1.10752691e+00, -1.76559471e-01, 2.04921820e-06, -7.49737179e-03, -1.04269540e-02, 
                                        3.61952370e-08, -9.60019557e+00, -1.25433200e+01, -3.16697240e-03, 6.04063434e+00, -2.36234755e-01, -1.27128770e-07, -5.91423830e-01, -3.47314386e-08, -7.77108894e-04, -1.83758536e-02, -1.89584563e-01, 1.76745863e-03, 1.08429244e-07, -1.18947761e-02, -5.49556582e-04, 4.03984868e-06, -7.98963107e-07, 8.66756613e-07, 2.71381004e-05, -3.89614418e-06, -4.67892884e-07, 3.11030459e-03, 4.80101427e-06, -3.53152511e-03, 1.50484916e+01, 1.46637387e+01, 1.37479574e-09, -8.07259516e-04, 1.28559140e+01, 5.91512085e+00, -2.00813625e-01, 
                                        9.38313418e-05, -1.55120343e-01, -8.91438504e-05, 3.12055261e-01, 2.85778729e-07, -1.44033727e-07, -1.75668296e-05, -1.61330781e-05, -1.32107096e-02, -1.49776879e-04, -3.59865093e-04, 2.68147489e-03, 1.54207356e-04, 8.32957884e-01, -1.87100044e-08, 2.36860609e-02, 1.18913391e-06, -1.93672636e-01, -1.86530104e-02, -9.98431150e-11, -1.10390211e+00, -1.30094217e+01, -1.15755666e+01, -4.32047055e-05, 5.33452413e-04, 1.39561332e-11, -1.29667900e-02, 7.41593409e-01, 2.45370857e-03, 2.00631345e-04, 2.32166274e-06, -2.91257026e-07, 
                                        -1.71838677e-04, 1.02135909e-07, -2.65456311e-03, 1.00946766e-07, 3.67319529e-04, -8.96608429e-07, 6.10268937e-06, -8.08933588e-10, 6.23259601e-01, -1.72753249e-05, -1.57391102e-02, -2.05761239e-07, 4.38671335e-08, 5.25111097e-04, 1.25285441e+01, 1.48192241e-06, 7.19880991e+00, -1.41556688e-04, -2.51362919e+00, -7.95536688e+00, -1.25125228e+01, 1.06597625e-02,
                                        -2.48728797e-08, -1.04578075e+01, -7.30495286e+00, 1.68686343e+01, 1.00449793e+01, 4.47533368e+00, 6.57446264e-02, -9.34506970e+00, -8.59056760e-01, 4.74698051e-01, -3.19128972e+00, -8.71488273e+00, -1.31997583e+01, 9.73555967e+00, -1.92787375e+01, -1.11493867e+01, -8.01254111e+00, 1.88225717e+01, -9.50835965e-10, -8.63422627e-04, -2.33724213e-01, -9.46339233e+00, -9.95915257e-01, -5.25865714e+00, -7.93216746e+00, -2.47030362e+00, 2.07958549e+00, 1.43403669e+00, 8.35978601e+00, 3.63464444e-14, -3.11424080e+00]) 

                # Define the dependent variable
                name_dep_var = crosssell_types_max
                # Say which covariates we are going to use
                name_covariates = full_covariates
                # take a subset of the number of periods
                df_periods  = dflist[4:10]  # use periods 5,6,7,8,9 and 10
                #Define number of segments
                n_segments = 5
                reg = 0.05 # Regularization term
                max_method = 'Nelder-Mead'
                run_cross_sell = True
                

            print(f"dependent variable: {name_dep_var}")
            print(f"covariates: {name_covariates}")
            print(f"number of periods: {len(df_periods)}")
            print(f"number of segments: {n_segments}")
            print(f"number of people: {len(dflist[0])}")
            outname = f"interpretparam_activity_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
            
            hmm = ht.HMM_eff(outdirec, outname,
                         df_periods, reg, max_method,
                         name_dep_var, 
                         name_covariates, covariates = True,
                         iterprint = True,
                         initparam = param_cross)

            
        #----------------------------------------------------------------------------
            
        # Now interpret & visualise the parameters 
        p_js, P_s_given_Y_Z, gamma_0, gamma_sr_0, gamma_sk_t, transition_probs = hmm.interpret_parameters(param_cross, n_segments)
        
        print("getting standard errors")
        hess_inv, dfSE = hmm.get_standard_errors(param_cross, n_segments)
        
        if run_cross_sell == True: # do we want to run the model for cross sell or activity
            tresholds = [0.2, 0.7]
            order_active_high_to_low = [0,1,2]
            t = 9 
            active_value_pd = pd.read_csv(f"{outdirec}/active_value.csv")
            active_value = active_value_pd.to_numpy()
            active_value = active_value[:,1]
            dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                      active_value, tresholds, 
                                                                                                      order_active_high_to_low)
            n_cross_sells = hmm.number_of_cross_sells(cross_sell_target, cross_sell_self, cross_sell_total)
             
        else:
            print("Calculating active value")
            active_value  = hmm.active_value(param_cross, n_segments, len(df_periods))
            active_value_df = pd.DataFrame(active_value) 

            utils.save_df_to_csv(active_value_df, outdirec, f"active_value", 
                             add_time = True )
            #active_value_df.to_csv(f"{outdirec}/active_value_t{t}.csv")

# =============================================================================
# SALDO PREDICTION
# =============================================================================

        if (saldopredict):
            print(f"****Create data for saldo prediction at {utils.get_time()}****")
            
            namesal = "final_df" # We use the full dataset with all complete IDs
            saldodflist = [pd.read_csv(f"{interdir}/{namesal}_2018Q1.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2018Q2.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2018Q3.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2018Q4.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2019Q1.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2019Q2.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2019Q3.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2019Q4.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2020Q1.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2020Q2.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2020Q3.csv"),
                        pd.read_csv(f"{interdir}/{namesal}_2020Q4.csv")]
        
            globalmin = np.inf    
            for i, df in enumerate(saldodflist):    
                df = saldodflist[i]
                df = additdata.aggregate_portfolio_types(df)
                saldodflist[i]= additdata.transform_variables(df)
                # Now also obtain a global minimum of all the datasets
                saldo = saldodflist[i]['saldototaal']
                globalmin = min(globalmin,min(saldo))
            
            print(f"overall minimum: {globalmin}")
               
            # Define which 'normal' variables to use in the dataset
            selection = activity_total # add activity vars
            
            # Define which dummy variables to use
            dummies = person_dummies # get experian characteristics
            dummies.extend(activity_dummies) # get activity status
            
            # Run the data creation
            saldo_name = f"saldopredict"
            predictdata = additdata.create_saldo_data(saldodflist, interdir,
                                            filename= saldo_name,
                                            select_variables = selection,
                                            dummy_variables = dummies,
                                            globalmin = globalmin)
            
            print(f"****Create saldo prediction model at {utils.get_time()}****")
            
            # get extra saldo 
            t = 10 # period for which we predict TODO make this a variable somewhere
            #minimum = 80000
            finergy_segment = "B04"
            
            predict_saldo = ps.predict_saldo(saldo_data = predictdata,
                                             df_time_series = saldodflist,
                                             interdir = interdir,
                                             )
            
            extra_saldo,  X_var_final, ols_final = ps.get_extra_saldo(cross_sell_total, globalmin, t, segment = finergy_segment)
            

# =============================================================================
# Evaluate output
# =============================================================================
      
        if run_cross_sell == True:    
          print("****Evaluating cross-sell targeting results****")
          t = len(df_periods)
          last_period = df_periods[t-1]
          testing_period = dflist[10] # period 10 or 11 can be used
          # tODO: make something so that the testing period becomes a variable??
         
          #---------------- GINI COEFFICIENT CALCULATION -------------------------
          
          print("Caclulating gini coefficient")
          # These describe product ownership yes/no in the new period
          prod_ownership_new = testing_period[crosssell_types_dummies]  
          # These describe product ownership yes/no in the previous period
          prod_ownership_old = last_period[crosssell_types_dummies]
          
          ginivec = hmm.calculate_gini(prod_ownership_new, prod_ownership_old, prod_own,
                          binary = True)
          print("Ginicoefficient for the binary ownership dummies")
          print(ginivec)
          
          # Now do it with the actual numbers of ownership
          prod_ownership_new = testing_period[crosssell_types_max]
          prod_ownership_old = last_period[crosssell_types_max]
          ginivec2 = hmm.calculateGini(prod_ownership_new, prod_ownership_old, prod_own,
                          binary = False)
          print("Ginicoefficient for the (non-binary) ownership variables")
          print(ginivec2)
          
          #--------------------- TP/ NP calculation -------------------------
          diffdata = additdata.get_difference_data(testing_period, last_period,
                                           select_variables = None,
                                           dummy_variables = None,
                                           select_no_decrease = False,
                                           globalmin = None)
            
          # These dummies describe whether an increase took place
          diffdata = diffdata["personid", "business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]  
          FPvec = []
          TPvec = []
          FNvec = []
          TNvec = []
          accuracyvec = []
          
          for i, column in enumerate(["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]):
            pred_labels = cross_sell_self[:,i]
            true_labels = diffdata[:,i] # check that this selects the right things
            
            # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
            
            TPvec.append( TP / (TP + FN))
            TNvec.append( TN / (TN + FP))
            FPvec.append( FP / (FP + TN))
            FNvec.append( FN / (FN + TP))
            
            accuracy = (TP+TN) / (TP+TN+FP+FN)
            sensitivity = TP / (TP+FN)
            specificity = TN/ (TN+FP)
            precision = TP / (TP+FP)
            accuracyvec.append(accuracy)
            
        print(f"Accuracy output: {accuracyvec}")


# =============================================================================
# Models testing
# =============================================================================

    LRtest = False
    if (LRtest) :
        
        print("****Doing LR comparison*****")
        def likelihood_ratio(llmin, llmax):
            return(2*(llmax-llmin))
        
        def calculateBIC(ll,k,n):
            return(k*np.log(n) - 2*ll)
        
        # Comparing 4 and 5 
        L1 = -2107.45952780524
        param1 = 105
        n1 =500
        BIC1 = calculateBIC(L1,param1,n1)
        print(f"BIC model 1: {BIC1}")
        
        L2 = -2050.340265791282
        param2 = 142
        n2 =500 
        BIC2 = calculateBIC(L2,param2,n2)
        print(f"BIC model 2: {BIC2}")
        
        if (BIC1<BIC2):
            print("Model 1 is better according to BIC")
        else:
            print("Model 2 is better according to BIC")
        
        LR = likelihood_ratio(L1,L2)
        p = chi2.sf(LR, param2-param1) # L2 has 1 DoF more than L1
        
        print('p: %.30f' % p) 
        if (p <0.05):
            print("Model 2 is significantly better according to LR test")
        else:
            print("Model 1 is significantly better according to LR test")
        
        # So 5 is significantly better than 4?
        
        
        
        

