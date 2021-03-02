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
    # indirec = "C:/Users/matth/OneDrive/Documenten/seminar 2021/data"
    # outdirec = "C:/Users/matth/OneDrive/Documenten/seminar 2021/output"
    # interdir = "C:/Users/matth/OneDrive/Documenten/seminar 2021/interdata"

    save_intermediate_results = False # Save the intermediate outputs
    # save_intermediate_results = True # Save the intermediate outputs
    print_information = True # Print things like frequency tables or not

    quarterly = True # In which period do we want to aggregate the data?
    #start_date = "2020-01-01" # From which moment onwards do we want to use
    start_date = "2018-01-01" # From which moment onwards do we want to use

    # the information in the dataset
    end_date = None # Until which moment do we want to use the information
    end_date = "2019-12-31" # Until which moment do we want to use the information
    subsample = False # Do we want to take a subsample
    #subsample = True # Do we want to take a subsample
    sample_size = 500 # The sample size
    finergy_segment = "B04" # The finergy segment that we want to be in the sample
    # e.g.: "B04" - otherwise make it None
    
    # How to save the final result
    if (finergy_segment != None):
        if subsample:
            final_name = f"final_df_fin{finergy_segment}_n{sample_size}"
        else:
            final_name = f"final_df_fin{finergy_segment}"
    else:
        if subsample:
            final_name = f"final_df_n{sample_size}"
        else:
            final_name = "final_df"
    
# =============================================================================
# DEFINE WHAT TO RUN
# =============================================================================
    
    cross_sec = False # Do we want to run the code for getting a single cross-sec
    time_series = False # Do we want to run the code for getting time series data
    transform = True # Transform & aggregate the data
    saldo_data = False # Do we want to create the dataset for predicting saldo
    visualize_data = True # make some graphs and figures
    
    run_hmm = False
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
        "aantalproducten_totaal"
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
    
       print("Doing a check that the ID columns are the same")
       for i, df in enumerate(dflist):    
           if (i==0): 
               dfold = dflist[i]
           else:
               dfnew = dflist[i]
               if (dfold["personid"].equals(dfnew["personid"])):
                   print("check")
               else:
                   print("noooooooooooo")
               dfold = dfnew
    
    
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
                                        filename= "saldopredict",
                                        select_variables = selection,
                                        dummy_variables = dummies)

    #-----------------------------------------------------------------
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")
    
    
# =============================================================================
# VISUALIZE DATA
# =============================================================================

    if visualize_data:
        print("*****Visualising data*****")
        df = dflist[0] # use the first time period
        visualize_variables = ["age_bins", "geslacht",
                               "hh_size",
                               "income",
                               "business_max",
                               "retail_max",
                               "joint_max",
                               "accountoverlay_max",
                               "activitystatus",
                               "saldototaal_bins"]   
        for var in visualize_variables:
            print(f"visualising {var}")
            DI.plotCategorical(df, var)
            counts = df[var].value_counts().rename_axis(var).to_frame("total count").reset_index(level=0)
            print(counts)

    
# =============================================================================
# RUN HMM MODEL
# =============================================================================
    
    if (transform):
         # MAKE ACTIVITY VARIABLES
        for i, df in enumerate(dflist):            
            df = dflist[i]
            dummies, activitynames =  additdata.make_dummies(df,
                                                 activity_dummies,
                                                 drop_first = True)
            df[activitynames] = dummies[activitynames]
        print("Dummy variables made:")
        print(activitynames)
        activity_variables = activitynames #activity status 1.0 is de base case
        activity_variables.extend(activity_total)
        
        #MAKE PERSONAL VARIABLES (ONLY INCOME, AGE, GENDER)
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
        base_cases = ['income_1.0','age_bins_(18, 30]','geslacht_Man']
                      #,'geslacht_Man(nen) en vrouw(en)']
        # Note: we should drop man(nen) en vrouw(en) as well!! Which means we treat these
        # occurrences as men
        personal_variables = [e for e in dummynames if e not in base_cases]


    if (run_hmm & transform):
        #---------------- SELECT VARIABLES ---------------
        print(f"****Defining variables to use at {utils.get_time()}****")

        if (run_cross_sell): 
            print("Running cross sell model")
            # Define the dependent variable
            name_dep_var = crosssell_types_max_nooverlay
            # Say which covariates we are going to use
            name_covariates = personal_variables
            # take a subset of the number of periods, just to test
            df_periods  = dflist[:5] # only use 5 periods for now?
            #Define number of segments
            n_segments = 3
            
        else: # run the activity model!
            print("Running activity model")
            # Define the dependent variable
            activity_dummies.extend(activity_total_dummies)
            name_dep_var = activity_dummies
            # Say which covariates we are going to use
            name_covariates = personal_variables
            # take a subset of the number of periods, just to test
            df_periods  = dflist[:8] # only use first 2 years
            #Define number of segments
            n_segments = 3
        
        #---------------- RUN THE HMM MODEL ---------------
        
        startmodel = utils.get_time()
        print(f"****Running HMM at {startmodel}****")
        print(f"dependent variable: {name_dep_var}")
        print(f"covariates: {name_covariates}")
        print(f"number of periods: {len(df_periods)}")
        print(f"number of segments: {n_segments}")
        print(f"number of people: {len(dflist[0])}")
        
        # Note: the input datasets have to be sorted / have equal ID columns!
        hmm = ht.HMM_eff(df_periods, name_dep_var, 
                                     name_covariates, covariates = True,
                                     iterprint = True)
        
        # Run the EM algorithm - max method can be Nelder-Mead or BFGS
        param_cross, alpha_cross, shapes_cross, hes = hmm.EM(n_segments, 
                                                             max_method = 'Nelder-Mead') 
       
        
        # Transform the output back to the specific parameter matrices
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(hmm, 
                                                                          n_segments, 
                                                                          param_cross, 
                                                                          shapes_cross)
        # Also print the Hessian?
        cov = - np.linalg.inv(hes)
        print(f"Covariance: {cov}")
    
        endmodel = utils.get_time()
        diff = utils.get_time_diff(startmodel,endmodel)
        print(f"HMM finished! Total time: {diff}")



# =============================================================================
# INTERPRET PARAMETERS
# =============================================================================

    if (transform & interpret):
        print("****Checking HMM output****")
        
        if (~run_hmm): # read in parameters if we have not run hmm
            print("-----Reading in existing parameters-----")
             
            source = "activityFinB04"
            #source = "3seg500n"
            if (source == "3seg500n"):
                # note: dataset used is "final_df_n500" _. make sure to change
                param_cross = np.array([ 9.57347300e+00,5.48433032e+00,1.12262507e+01,1.62844971e+01
                ,-4.24443740e+00,-9.38509250e-13,-2.93732866e+00,-1.87993462e+00
                ,-2.65321128e+00,1.46505313e-20,2.96302734e+02,1.09186362e+01
                ,1.02184573e+01,6.47918756e+00,1.15696666e+01,1.64926188e+01
                ,-3.90221106e+00,-1.18444012e-15,-3.90183094e+00,-2.60783116e+00
                ,-3.03147808e+00,5.73839349e+01,1.88850731e-12,1.10490188e+01
                ,2.00427593e+01,1.38660471e-25,2.54295940e-24,2.54256611e-30
                ,2.75082911e+02,-3.36123056e+01,2.71594678e-18,-9.76973359e-23
                ,9.64370896e+00,-6.91851040e-25,1.61978835e-18,-1.47880050e+01
                ,-1.45612764e+01,-8.18246359e+00,5.44864002e-19,5.90501218e+02
                ,5.84185133e-01,1.73384950e+00,1.04830966e+00,1.02836248e+01
                ,-6.53726714e+01,5.14140979e-21,1.59860159e-01,2.01874672e-21
                ,7.57429109e-23,-9.65927404e-19,-4.26094993e-15,-1.97230447e-21
                ,5.89207048e+00,9.05672539e+00,-1.46727080e+01,-7.31147855e-01
                ,6.73366209e+00,5.81941813e+00,3.29402028e-17,-9.26004012e+01
                ,-2.23678235e+02,1.17238854e+02,3.56124603e-20,5.68453198e+00
                ,-5.27674988e+00,-1.01674686e+01,1.17725048e+01,1.65711107e+00
                ,-7.05901628e+00,-1.06283721e+01])
                
                # Define the parameters (need to be the same as what was used to
                # produce the output!)
                df_periods  = dflist[:5] 
                name_dep_var = crosssell_types_max_nooverlay
                name_covariates = personal_variables
                n_segments = 3
                
            if (source == "activityFinB04"):
                
                # note: dataset used is "final_df_finB04"
                param_cross = np.array([ 1.09607100e-01,9.08286566e-03,-6.36999543e-01,-2.96600310e-01
                ,-3.12565519e-01,6.46833377e+05,3.84478055e-01,3.22616130e-01
                ,3.34262554e-01,-3.98285606e+01,5.18491848e-01,1.64299266e-01
                ,-4.32951294e-02,9.28187258e-01,3.17080787e-01,1.35084244e-01
                ,9.72483796e-02,-4.39334060e+07,2.01594729e-01,-2.19203431e-01
                ,-6.51141160e-01,-1.74737789e+03,1.20225968e+00,1.71622738e-01
                ,3.63044788e+00,8.74362292e+00,-3.30587117e+00,1.18486016e+00
                ,1.42813089e+01,-1.76406965e+01,1.26248695e+00,3.35600755e-02
                ,-9.33111839e-02,-1.16092695e-01,-6.89555430e-02,3.65032170e-01
                ,3.40766501e-02,1.55316201e-01,-2.46821210e-01,-4.59052570e-01
                ,8.64342959e-02,1.35125683e+00,-5.50664918e-01,-1.27545667e+00
                ,-9.28133544e-01,-2.50734713e+04,-7.18313052e-01,-8.74515160e-01
                ,-1.43949098e+00,-1.94122989e+01,4.76542140e-01,1.80622664e-01
                ,2.55926773e+01,2.73330647e+01,3.23366097e+01,4.38701084e+00
                ,6.12311842e+00,5.65507768e+00,1.16063458e+01,-1.02975081e+04
                ,-1.55988692e+06,1.33866191e+01,1.36217551e+01,1.99016626e+01
                ,1.21420722e+01,2.22883871e+01,1.43737298e+01,1.48039032e+01
                ,1.14224853e+01,-1.50407630e+00,-5.59738143e+00,-2.64778064e+00
                ,-9.17047760e+00])
               
                # Define the parameters (need to be the same as what was used to
                # produce the output!)
                df_periods  = dflist[:8] 
                activity_dummies.extend(activity_total_dummies)
                name_dep_var = activity_dummies
                name_covariates = personal_variables
                n_segments = 3
            
            
            print(f"dependent variable: {name_dep_var}")
            print(f"covariates: {name_covariates}")
            print(f"number of periods: {len(df_periods)}")
            print(f"number of segments: {n_segments}")
            print(f"number of people: {len(dflist[0])}")
            
            
            hmm = ht.HMM_eff(df_periods, name_dep_var, name_covariates, 
                             covariates = True, iterprint = True)
            
        # Now interpret & visualise the parameters 
        hmm.interpret_parameters(param_cross, n_segments)

        # # See what the results are in terms of probabilities, for the first period
        # Y = hmm.list_Y[0]
        # Z = hmm.list_Z[0]
        # p_js = ef.prob_p_js(hmm, param_cross, shapes_cross, n_segments)
        # P_y_given_S = ef.prob_P_y_given_s(hmm, Y, p_js, n_segments)
        # P_s_given_Z = ef.prob_P_s_given_Z(hmm, param_cross, shapes_cross, Z, n_segments)
        # P_s_given_r = ef.prob_P_s_given_r(hmm, param_cross, shapes_cross, Z, n_segments)

    
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

        
        
        
        
        
        
        

