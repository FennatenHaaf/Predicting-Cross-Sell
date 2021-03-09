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
            saldo_name =  f"saldopredict_fin{finergy_segment}_n{sample_size}"
        else:
            final_name = f"final_df_fin{finergy_segment}"
            saldo_name = f"saldopredict_fin{finergy_segment}"
    else:
        if subsample:
            final_name = f"final_df_n{sample_size}"
            saldo_name = f"saldopredict_n{sample_size}"
        else:
            final_name = "final_df"
            saldo_name = f"saldopredict"
# =============================================================================
# DEFINE WHAT TO RUN
# =============================================================================
    
    cross_sec = False # Do we want to run the code for getting a single cross-sec
    time_series = False # Do we want to run the code for getting time series data
    transform = True # Transform & aggregate the data
    saldo_data = False # Do we want to create the dataset for predicting saldo
    visualize_data = False # make some graphs and figures
    
    run_hmm = True
    run_cross_sell = True # do we want to run the model for cross sell or activity
    interpret = True #Do we want to interpret variables

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
        df_cross, cross_date = processor.create_base_cross_section(date_string="2020-12-31", 
                            next_period = False, outname = "cross_experian")
        # Now aggregate all of the information per portfolio
        df_cross_link = processor.create_cross_section_perportfolio(df_cross, cross_date, 
                                              outname = "df_cross_portfoliolink")
        # Aggregate all of the portfolio information per person ID
        df_out = processor.create_cross_section_perperson(df_cross, df_cross_link,
                                            cross_date, outname = final_name)
    
    #----------------MAKE TIME SERIES DATASETS-------------------
    if time_series:
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
    if transform:    
       print(f"****Transforming datasets & adding some variables at {utils.get_time()}****")
       
       for i, df in enumerate(dflist):    
            df = dflist[i]
            df = additdata.aggregate_portfolio_types(df)
            dflist[i]= additdata.transform_variables(df)
    
        
       print("Doing a check that the ID columns are the same, and getting minimum saldo")
       
       globalmin = np.inf    
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
               
           # Now also obtain a global minimum of all the datasets
           saldo = dfold['saldototaal']
           globalmin = min(globalmin,saldo)
           
    
    
    #--------------- GET DATA FOR REGRESSION ON SALDO ------------------
    if (saldo_data & transform):
        print(f"****Create data for saldo prediction at {utils.get_time()}****")
        
        # Define which 'normal' variables to use in the dataset
        selection = activity_total # add activity vars
        
        # Define which dummy variables to use
        dummies = person_dummies # get experian characteristics
        dummies.extend(activity_dummies) # get activity status
        
        # Run the data creation
        predictdata = additdata.create_saldo_data(dflist, interdir,
                                        filename= saldo_name,
                                        select_variables = selection,
                                        dummy_variables = dummies,
                                        globalmin = globalmin)

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
    
    if (transform):
        
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
        # Note: we should drop man(nen) en vrouw(en) as well!! Which means we treat these
        # occurrences as men
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



    if (run_hmm & transform):
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
            n_segments = 4
            reg = 0.05 # Regularization term
            
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
            
            outname = f"activity_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
        
        
        #---------------- Enter initial parameters ---------------

        #initial_param = None 
        # initial_param = np.array([ 4.00279803e-01, 1.55148857e+00, 3.01075360e-01,
        #                           3.61195130e-01, 2.44142717e-01, 1.97377519e-01, -1.21491736e-01,
        #                           -3.76051007e-01, -1.09935554e+00, 1.69131815e-01, -3.71756994e-01,
        #                           1.03314763e+00, -4.57133282e-01, -1.86250226e-01, -1.82250901e-01,
        #                           6.10375682e-01, 5.35357171e-01, 4.37577944e-01, 1.47101548e-01,
        #                           1.46280268e-01, 8.81679443e+00, 1.38150859e-01, -9.14823349e+00,
        #                           5.61651293e+00, 4.06077770e+00, -2.47700418e+00, 1.12583167e+00,
        #                           1.23619704e-01, -7.40264237e-01, -6.67140557e-01, 1.23417985e-02,
        #                           -5.47161920e-01, 4.94438786e-04, 5.38637909e-02, 4.70217916e-01,
        #                           -8.63527345e-04, -1.02960062e+00, -1.12182604e+00, -1.33946194e+00,
        #                           4.75096107e-01, 3.23766601e-01, 2.61892254e-01, 1.10839026e+00,
        #                           -9.81763889e-02, -1.61596020e-03, -8.42801686e-01, 8.86826217e+00,
        #                           1.08790730e+01, 1.54395264e+01, 9.31398860e+00, 1.86251543e+01,
        #                           -4.15505145e-04, 2.08954057e+00, 7.48087332e+00, 3.96424280e+00,
        #                           5.74881298e+00, 6.32078146e+00, 8.66508230e+00, 8.10629803e+00,
        #                           8.41172505e+00, 4.28800496e+00, -1.71701327e+00, -5.80770925e+00,
        #                           -3.19102183e+00, -8.70860889e+00]) # paste parameters here!

        initial_param = np.array(
            [3.26975644e-03, -2.40317092e-02, -9.64635527e-01, -7.99715829e-05, -4.08861564e-01, 1.87508664e-03, -1.16239502e-15,
             -4.03948601e+00, 1.49513616e+00, -3.07920354e-04, 3.66774819e-03, -2.09372586e-04, -8.69873829e-04, -1.48926750e+00,
             -8.46496314e-04, -1.22483181e-01, -2.22631380e-03, -1.56075570e-04, 1.11211614e-08, 2.42772532e+00, 5.79780134e-03,
             -9.27133928e-07, -1.17319840e-03, 2.75042091e+01, -3.09903359e-03, 2.32471348e+01, 1.07754604e-10, 6.50176285e+00,
             -1.21245734e+00, 3.91548382e-01, 1.95805777e-03, 3.23247284e+00, 2.23115839e-13, 1.54397245e+00, -3.60995311e+00,
             7.53117655e-02, -2.80438850e+00, -2.67182312e-03, -7.99397746e-01, 4.11234483e-02, -2.63025526e-08, 2.16461929e+00,
             -2.33483271e-02, 1.12977807e-05, -2.19936842e+00, 1.47404935e-03, 4.27837400e+00, -1.26202716e-06, 7.23315405e-05,
             1.72050647e+01, -1.28078229e-05, -8.92082474e-13, 1.05901360e+01, -7.28217535e-01, 1.50994118e+01, -1.24385458e+00,
             2.53981147e-01, -1.34317130e-08, 7.86300137e-01, -4.21471881e-04, 1.40080429e-03, -1.31767923e+00, 1.77449599e+00,
             -1.44069012e-03, -3.14933693e-02, 1.38118973e-03, -6.83812845e-03, 1.78605088e-05, 1.43869511e+00, 1.74619789e-03,
             3.04520909e-03, -1.21680976e-05, 7.06444309e-04, 3.35779129e+00, 1.72409754e-02, -3.85068843e-03, -5.53459945e-10,
             -3.83303297e-10, -1.24752885e+01, -5.08142648e-11, -1.60177974e+01, -6.45165972e-04, -1.59380841e-05, -1.08801646e+00,
             9.90715045e-03, -2.15771776e-04, -3.72761324e-04, 4.45573947e-04, -1.36759550e-02, 2.76295958e+00, 3.75503062e+00,
             9.64761128e+00, -8.62688027e-03, 1.54967724e-03, 1.08700238e+00, 1.24622331e-05, -1.45073060e-07, 5.94036221e-07,
             -7.87214934e-04, 4.96893618e-02, 1.67953198e-04, 9.94175648e-05, -5.54807079e-03, 1.36761089e+00, 1.26137997e-02,
             1.09163582e-02, 9.00714640e-07, 1.13287807e-07, -3.15465958e-06, 1.32511959e-05, 1.72769710e-03, -2.22836567e-03,
             4.53046474e-04, -1.61912077e-03, 5.10590273e-03, 1.39395265e-03, -8.28089514e-04, 2.28383022e-03, -1.48375240e-03,
             1.95765837e+00, -5.09304918e-02, 1.94787780e-02, -8.86134299e-10, 7.87621147e-04, 2.05858558e-04, -2.04032120e-07,
             -2.35453766e-08, -4.80810427e-11, -7.59097258e-05, -1.03238386e-02, -1.54895070e-08, 2.43732755e-06, -1.87795838e-06,
             6.51949534e-06, 5.24193687e-07, 3.58067470e-07, 5.42142160e-02, -1.80587195e-02, -3.35873401e-05, 5.03077389e-06,
             1.58381382e+01, 1.70035841e+01, 3.92173817e-01, 1.39646586e+01, 1.34433240e+01, 2.07200025e+00, 8.43392503e-08,
             1.23339186e-04, -2.40983941e-05, 1.35409105e-06, 6.46144072e-01, 4.01660761e-01, -3.03942702e-04, -5.62646404e-05,
             -3.91320620e-04, 2.97156016e-04, -7.34814578e-02, -1.39959075e-06, -2.14185427e-04, -4.95773618e-03, 8.15171598e-02,
             2.24425846e-04, 4.77742883e-05, 3.95535903e-04, 1.72742681e-04, -9.24815277e-06, 3.27807089e-05, -3.52317427e-11,
             -1.26344523e+01, -2.11228715e-11, 9.79762369e-07, 9.59012874e+00, 8.42221811e+00, -7.22296960e+00, -5.29501139e-04,
             -5.11804338e-01, 8.10802170e-01, -6.37867341e+00, 1.05922265e+01, 1.26280278e+01, -3.37556489e-05, 9.97793225e-02,
             -6.52814693e-01, 8.71671323e-01, -6.75100973e-01, -1.06974518e+01, -7.26386407e+00, -1.88230606e+01, 1.32063815e+00,
             -3.43993786e-01, -9.83366798e+00, -9.59646876e+00, -8.62310743e-01, -4.26963613e+00, 7.29571622e+00, 1.37778943e+00,
             -3.61281998e-01, -1.73630985e+00, -1.97277530e+00])


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
                         df_periods, name_dep_var, 
                         name_covariates, covariates = True,
                         iterprint = True,
                         initparam = initial_param)

        # Run the EM algorithm - max method can be Nelder-Mead or BFGS
        param_cross,alpha_cross,beta_cross,shapes_cross,hess_inv = hmm.EM(n_segments, 
                                                             max_method = 'Nelder-Mead',
                                                             reg_term = reg,
                                                             random_starting_points = True)  

        # Transform the output back to the specific parameter matrices
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(hmm, 
                                                                          n_segments, 
                                                                          param_cross, 
                                                                          shapes_cross)
        # Also print the covariance
        # cov = np.linalg.inv(-hes)
        # print(f"Covariance: {cov}")
    
        endmodel = utils.get_time()
        diff = utils.get_time_diff(startmodel,endmodel)
        print(f"HMM finished! Total time: {diff}")



# =============================================================================
# INTERPRET PARAMETERS
# =============================================================================

    if (transform & interpret):
        print("****Checking HMM output****")
        
        if (not run_hmm): # read in parameters if we have not run hmm
            print("-----Reading in existing parameters-----")
             
            source = "finalActivity"
                          
            if (source == "finalActivity"):
                
                # note: dataset used is "final_df_finB04"
                param_cross = np.array([ 3.99872551e-01, 1.55174007e+00, 3.01236578e-01, 3.60653223e-01,
                2.44391119e-01, 1.97533254e-01, -1.21325482e-01, -3.75900089e-01, -1.09597705e+00, 1.69133466e-01, 
                -3.71668823e-01, 1.03317680e+00, -4.56861382e-01, -1.86047755e-01, -1.82113764e-01, 6.10210129e-01,
                5.35256621e-01, 4.37430282e-01, 1.46450243e-01, 1.46182996e-01, 8.81644633e+00, 1.38062593e-01, 
                -9.14871380e+00, 5.61629100e+00, 4.06053603e+00, -2.47751929e+00, 1.12557711e+00, 1.23822825e-01, 
                -7.39454264e-01, -6.66564174e-01, 1.24907110e-02, -5.47088303e-01, 5.10578630e-04, 5.36773181e-02,
                4.70270887e-01, -8.83672811e-04, -1.02892266e+00, -1.12122780e+00, -1.33884713e+00, 4.74891845e-01,
                3.23462965e-01, 2.61437448e-01, 1.10933134e+00, -9.80237698e-02, -1.57410914e-03, -8.42882097e-01, 
                8.86916524e+00, 1.08789822e+01, 1.54395082e+01, 9.31348666e+00, 1.86245690e+01, -4.32810937e-04, 
                2.08956735e+00, 7.48082274e+00, 3.96433823e+00, 5.74870230e+00, 6.32090251e+00, 8.66492623e+00, 
                8.10626061e+00, 8.41162052e+00, 4.28823154e+00, -1.71701599e+00, -5.80767319e+00, -3.19105463e+00, 
                -8.70848005e+00]) 

                # Define the dependent variable
                activity_dummies.extend(activity_total_dummies)
                name_dep_var = activity_dummies # activitystatus, logins, transactions
                # Say which covariates we are going to use
                name_covariates = personal_variables # income, age, gender
                # take a subset of the number of periods 
                df_periods  = dflist[4:10] # use periods 5,6,7,8,9 and 10
                #Define number of segments
                n_segments = 4
                reg = 0.1 # Regularization term
                run_cross_sell = False
            
            print(f"dependent variable: {name_dep_var}")
            print(f"covariates: {name_covariates}")
            print(f"number of periods: {len(df_periods)}")
            print(f"number of segments: {n_segments}")
            print(f"number of people: {len(dflist[0])}")
            outname = "interpretparam"
            
            hmm = ht.HMM_eff(outdirec, outname,
                         df_periods, name_dep_var, 
                         name_covariates, covariates = True,
                         iterprint = True,
                         initparam = param_cross)

            
        #----------------------------------------------------------------------------
            
        # Now interpret & visualise the parameters 
        p_js, P_s_given_Y_Z, gamma_0, gamma_sr_0, gamma_sk_t, transition_probs = hmm.interpret_parameters(param_cross, n_segments)
        
        if run_cross_sell == True: # do we want to run the model for cross sell or activity
            tresholds = [0.2, 0.7]
            order_active_high_to_low = [0,1,2]
            t = len(df_periods)
            active_value_pd = pd.read_csv(f"{outdirec}/active_value_t{t}.csv")
            active_value = active_value_pd.to_numpy()
            active_value = active_value[:,1]
            dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                      active_value, tresholds, 
                                                                                                      order_active_high_to_low)
            n_cross_sells = hmm.number_of_cross_sells(cross_sell_target, cross_sell_self, cross_sell_total)
             
        else:

            t = len(df_periods)
            active_value  = hmm.active_value(param_cross, n_segments, t)
            active_value_df = pd.DataFrame(active_value) 

            active_value_df.to_csv(f"{outdirec}/active_value_t{t}.csv")

            #t = 10
            #active_value  = hmm.active_value(param_cross, n_segments, t)
            #active_value_df = pd.DataFrame(active_value) 

            #active_value_df.to_csv(f"{outdirec}/active_value.csv")

        # get extra saldo 
        t = 10
        #minimum = 80000
        finergy_segment = "B04"
        
        predict_saldo = ps.predict_saldo()
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
          
          #TODO: right now this is only for 1 or 0 whether they own the product or not!!
          # These describe product ownership yes/no in the new period
          prod_ownership_new = testing_period[crosssell_types_dummies]
          
          # These describe product ownership yes/no in the previous period
          prod_ownership_old = last_period[crosssell_types_dummies]
          
          
          Gini = []
          
          for i, column in enumerate(["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]):
              
            # Number of households who did NOT have product
            n_j = len(prod_ownership_old[:,i]==0) 
            
            # Percentage of those households who now do own the product
            select = (prod_ownership_old[:,i]==0)
            change = prod_ownership_new.loc[select,i] # todo check that this selects the right thing
            mu_j = (sum(change) / len(change))*100 # percentage that is 1
            
            # Ranked probabilities - 
            # We want the person with the highest probability to get the lowest rank
            probranks = prod_own[:,i].rank(method='max', ascending = False)
            
            sumrank = 0
            for j in range(0,len(testing_period)):
                sumrank += probranks[j] * prod_ownership_new[j,i]
              
                
            Ginij = 1 + (1/n_j) - ( 2 / ( (n_j**2)*mu_j  ) )*sumrank 
            Gini.append(Ginij)
              
         
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
          
          for i, column in enumerate(["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]):
            pred_labels = cross_sell_self[:,i]
            true_labels = diffdata[:,i] # check that this selects the right things
            
            # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
            TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
            TPvec.append(TP)
            # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
            TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
            TNvec.append(TN)
            # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
            FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
            FPvec.append(FP)
            # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
            FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
            FNvec.append(FN)
            
        # TODO: make accuracy score out of these

       


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

        
        
        
        
        
        
        

