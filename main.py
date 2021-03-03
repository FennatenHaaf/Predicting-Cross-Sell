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
    #indirec = "C:/Users/matth/OneDrive/Documenten/seminar 2021/data"
    #outdirec = "C:/Users/matth/OneDrive/Documenten/seminar 2021/output"
    #interdir = "C:/Users/matth/OneDrive/Documenten/seminar 2021/interdata"

    save_intermediate_results = False # Save the intermediate outputs
    # save_intermediate_results = True # Save the intermediate outputs
    print_information = False # Print things like frequency tables or not

    quarterly = True # In which period do we want to aggregate the data?
    #start_date = "2020-01-01" # From which moment onwards do we want to use
    start_date = "2018-01-01" # From which moment onwards do we want to use

    # the information in the dataset
    end_date = None # Until which moment do we want to use the information
    end_date = "2020-12-31" # Until which moment do we want to use the information
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
    visualize_data = False # make some graphs and figures
    
    run_hmm = False
    run_cross_sell = False # do we want to run the model for cross sell or activity
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
    
       print("Doing a check that the ID columns are the same")
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
                               "saldototaal_bins",
                               "SBIsectorName",
                               "businessAgeInYears_bins",
                               "businessType"]   
        for var in visualize_variables:
            print(f"visualising {var}")
            DI.plotCategorical(df, var)
            counts = df[var].value_counts().rename_axis(var).to_frame("total count").reset_index(level=0)
            print(counts)

    
# =============================================================================
# RUN HMM MODEL
# =============================================================================
    
    if (transform):
         #----------------MAKE ACTIVITY VARIABLES -----------------------
        # for i, df in enumerate(dflist):            
        #     df = dflist[i]
        #     dummies, activitynames =  additdata.make_dummies(df,
        #                                          activity_dummies,
        #                                          drop_first = True)
        #     df[activitynames] = dummies[activitynames]
        # print("Dummy variables made:")
        # print(activitynames)
        # activity_variables = activitynames #activity status 1.0 is de base case
        # activity_variables.extend(activity_total)
        
        #-------MAKE PERSONAL VARIABLES (ONLY INCOME, AGE, GENDER)-------------
        # we don't use all experian variables yet
        #TODO remove this once we do not need to interpret the parameters at 
        # the bottom anymore
        
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
        
        
        #---------MAKE PERSONAL VARIABLES (ONLY INCOME, AGE, GENDER) - FIXED-------
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
        base_cases = ['income_1.0','age_bins_(18, 30]','geslacht_Man'
                      ,'geslacht_Man(nen) en vrouw(en)','age_bins_(0, 18]']
        # Note: we should drop man(nen) en vrouw(en) as well!! Which means we treat these
        # occurrences as men
        personal_variables2 = [e for e in dummynames if e not in base_cases]

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
                      'hh_size_2.0','saldototaal_bins_(0,100]',
                      ]
        # Note: we should drop man(nen) en vrouw(en) as well!! Which means we treat these
        # occurrences as men
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
            # take a subset of the number of periods, just to test
            df_periods  = dflist[:5] # only use 5 periods for now?
            #Define number of segments
            n_segments = 5
            
        else: # run the activity model!
            print("Running activity model")
            # Define the dependent variable
            activity_dummies.extend(activity_total_dummies)
            name_dep_var = activity_dummies
            # Say which covariates we are going to use
            name_covariates = personal_variables2
            # take a subset of the number of periods, just to test
            df_periods  = dflist # only use first 2 years
            #Define number of segments
            n_segments = 3
        
        #---------------- RUN THE HMM MODEL ---------------
        
        reg = 0.05 # Regularization term - TODO move to top maybe
        
        startmodel = utils.get_time()
        print(f"****Running HMM at {startmodel}****")
        print(f"dependent variable: {name_dep_var}")
        print(f"covariates: {name_covariates}")
        print(f"number of periods: {len(df_periods)}")
        print(f"number of segments: {n_segments}")
        print(f"number of people: {len(dflist[0])}")
        print(f"regularization term: {reg}")
        
        # Note: the input datasets have to be sorted / have equal ID columns!
        hmm = ht.HMM_eff(df_periods, name_dep_var, 
                                     name_covariates, covariates = True,
                                     iterprint = True)
        
        # Run the EM algorithm - max method can be Nelder-Mead or BFGS
        param_cross, alpha_cross, beta_cross, shapes_cross, hes = hmm.EM(n_segments, 
                                                             max_method = 'Nelder-Mead',
                                                             reg_term = reg,
                                                             random_starting_points = True)  

        # Transform the output back to the specific parameter matrices
        gamma_0, gamma_sr_0, gamma_sk_t, beta = ef.param_list_to_matrices(hmm, 
                                                                          n_segments, 
                                                                          param_cross, 
                                                                          shapes_cross)
        # Also print the Hessian?
        cov = np.linalg.inv(-hes)
        print(f"Covariance: {cov}")
    
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
             
            #source = "activityFinB04"
            #source = "3seg500n"
            #source = "finalActivity"
            source = "crosssell20it"
            
              # if (source == "3seg500n"):
            #     # note: dataset used is "final_df_n500" _. make sure to change
            #     param_cross = np.array([ 9.57347300e+00,5.48433032e+00,1.12262507e+01,1.62844971e+01
            #     ,-4.24443740e+00,-9.38509250e-13,-2.93732866e+00,-1.87993462e+00
            #     ,-2.65321128e+00,1.46505313e-20,2.96302734e+02,1.09186362e+01
            #     ,1.02184573e+01,6.47918756e+00,1.15696666e+01,1.64926188e+01
            #     ,-3.90221106e+00,-1.18444012e-15,-3.90183094e+00,-2.60783116e+00
            #     ,-3.03147808e+00,5.73839349e+01,1.88850731e-12,1.10490188e+01
            #     ,2.00427593e+01,1.38660471e-25,2.54295940e-24,2.54256611e-30
            #     ,2.75082911e+02,-3.36123056e+01,2.71594678e-18,-9.76973359e-23
            #     ,9.64370896e+00,-6.91851040e-25,1.61978835e-18,-1.47880050e+01
            #     ,-1.45612764e+01,-8.18246359e+00,5.44864002e-19,5.90501218e+02
            #     ,5.84185133e-01,1.73384950e+00,1.04830966e+00,1.02836248e+01
            #     ,-6.53726714e+01,5.14140979e-21,1.59860159e-01,2.01874672e-21
            #     ,7.57429109e-23,-9.65927404e-19,-4.26094993e-15,-1.97230447e-21
            #     ,5.89207048e+00,9.05672539e+00,-1.46727080e+01,-7.31147855e-01
            #     ,6.73366209e+00,5.81941813e+00,3.29402028e-17,-9.26004012e+01
            #     ,-2.23678235e+02,1.17238854e+02,3.56124603e-20,5.68453198e+00
            #     ,-5.27674988e+00,-1.01674686e+01,1.17725048e+01,1.65711107e+00
            #     ,-7.05901628e+00,-1.06283721e+01])
                

            #     # Define the parameters (need to be the same as what was used to
            #     # produce the output!)
            #     df_periods  = dflist[:5] 
            #     name_dep_var = crosssell_types_max_nooverlay
            #     name_covariates = personal_variables
            #     n_segments = 3
            #     run_cross_sell = True

                
            # if (source == "activityFinB04"):
                
            #     # note: dataset used is "final_df_finB04"
            #     param_cross = np.array([ 1.09607100e-01,9.08286566e-03,-6.36999543e-01,-2.96600310e-01
            #     ,-3.12565519e-01,6.46833377e+05,3.84478055e-01,3.22616130e-01
            #     ,3.34262554e-01,-3.98285606e+01,5.18491848e-01,1.64299266e-01
            #     ,-4.32951294e-02,9.28187258e-01,3.17080787e-01,1.35084244e-01
            #     ,9.72483796e-02,-4.39334060e+07,2.01594729e-01,-2.19203431e-01
            #     ,-6.51141160e-01,-1.74737789e+03,1.20225968e+00,1.71622738e-01
            #     ,3.63044788e+00,8.74362292e+00,-3.30587117e+00,1.18486016e+00
            #     ,1.42813089e+01,-1.76406965e+01,1.26248695e+00,3.35600755e-02
            #     ,-9.33111839e-02,-1.16092695e-01,-6.89555430e-02,3.65032170e-01
            #     ,3.40766501e-02,1.55316201e-01,-2.46821210e-01,-4.59052570e-01
            #     ,8.64342959e-02,1.35125683e+00,-5.50664918e-01,-1.27545667e+00
            #     ,-9.28133544e-01,-2.50734713e+04,-7.18313052e-01,-8.74515160e-01
            #     ,-1.43949098e+00,-1.94122989e+01,4.76542140e-01,1.80622664e-01
            #     ,2.55926773e+01,2.73330647e+01,3.23366097e+01,4.38701084e+00
            #     ,6.12311842e+00,5.65507768e+00,1.16063458e+01,-1.02975081e+04
            #     ,-1.55988692e+06,1.33866191e+01,1.36217551e+01,1.99016626e+01
            #     ,1.21420722e+01,2.22883871e+01,1.43737298e+01,1.48039032e+01
            #     ,1.14224853e+01,-1.50407630e+00,-5.59738143e+00,-2.64778064e+00
            #     ,-9.17047760e+00])

            #     # Define the parameters (need to be the same as what was used to
            #     # produce the output!)
            #     df_periods  = dflist[:8] 
            #     activity_dummies.extend(activity_total_dummies)
            #     name_dep_var = activity_dummies
            #     name_covariates = personal_variables
            #     n_segments = 3
            #     run_cross_sell = True

            if (source == "crosssell20it"):
                param_cross = np.array([-1.59851179e-02,1.71030808e+00,6.39402247e-01,2.49854672e-05
                    ,6.37274643e-04,-1.25648848e-05,2.66934066e+00,2.28449489e+00
                    ,1.15084559e+00,7.27480512e-01,2.07551964e-01,1.29314579e+00
                    ,1.78496771e-02,-7.60715405e-02,-2.95724843e-02,-1.24139216e+00
                    ,-2.51956674e-08,-7.92760761e-04,-5.22163210e-01,-3.26633870e-05
                    ,-6.99474006e-02,-1.95060667e+00,-8.42691201e-01,5.98210010e-02
                    ,1.18145579e-01,7.30888229e+00,1.61191956e+01,1.20526759e+01
                    ,1.63685906e-07,1.15789390e-01,2.91538575e+00,1.39406783e+00
                    ,1.22888521e+00,2.87822057e+00,5.00824507e-01,-2.29663679e-04
                    ,3.17442791e+00,1.57042555e+00,-1.92714425e-04,-6.20987921e-05
                    ,1.70482866e-05,2.05838103e+00,1.86469347e-01,9.64964479e-01
                    ,-5.76578019e-02,-3.26836738e-01,2.01780900e-01,1.77845284e-04
                    ,7.07323952e-01,-2.78108046e-04,9.87071049e-01,7.95691506e-04
                    ,7.28746083e-01,5.19258202e-01,-1.31792271e+01,8.47831781e-04
                    ,-3.85743466e-06,6.24895447e-03,-1.23297159e+01,-1.41671249e-01
                    ,-4.04029303e+00,2.27528916e+00,8.69272704e-01,-4.33499952e-02
                    ,-1.86002427e-03,4.92850215e-04,2.37719427e+00,1.92023629e+00
                    ,2.32374955e-04,-1.19074506e-03,-6.10895985e-01,7.05796805e-01
                    ,3.46861391e-03,2.25368955e-04,2.00665685e-05,2.59505701e-02
                    ,7.22198872e-02,2.73346752e-05,1.02547840e-01,-2.34353443e-03
                    ,1.97312619e-03,-1.50788589e+00,-1.44991444e-02,1.53894502e+00
                    ,-3.16725064e-03,-6.15002162e-02,1.06277123e+01,5.71027180e-01
                    ,-3.42884546e+00,1.42885874e+01,6.98032897e+00,7.18144178e-01
                    ,1.01166376e+00,3.59934658e+00,1.25868002e+00,1.22215240e+00
                    ,3.98574504e+00,1.02799279e+00,-1.77844768e-03,-3.26125830e-02
                    ,-3.03947913e-03,2.18184442e+00,-5.35758646e-04,8.14437188e-01
                    ,-1.28497396e-02,1.91083582e-01,-2.90486709e-03,4.47856698e-02
                    ,4.42525754e-03,-2.88698527e-07,9.77132422e-01,-6.88246443e-01
                    ,-1.14882509e-02,3.00866322e-02,-1.48985785e+01,-2.73411534e+00
                    ,4.66480570e-05,-2.03316532e-02,-1.22165777e+01,3.23188631e-01
                    ,7.34170098e+00,1.41936874e+00,-1.09368022e-03,-2.76938797e+00
                    ,-4.19518511e+00,3.64198001e+00,9.61916893e+00,-8.51234791e-02
                    ,2.34250356e-04,-5.29230278e+00,-8.27572572e-06,-7.18346872e+00
                    ,1.54129427e-11,-1.13570682e+01,-6.34467592e+00,8.25680498e+00
                    ,5.41083201e+00,-1.08443326e-01,7.20648648e+00,5.47348755e-05
                    ,1.07808567e-06,2.45605693e-02,1.35203447e-02,-1.22895070e+00
                    ,-6.68093017e-03,1.75393764e-02,-3.76584501e-04,8.63335155e-06
                    ,-2.09674012e+00,-4.43273898e-03,2.55332268e-02,2.49418954e-05
                    ,-4.05852249e-01,-2.91817659e-03,1.60909701e-02,-4.44620301e-01
                    ,1.03753851e-01,-3.81521254e-04,-1.77347395e-03,-2.60120412e-01
                    ,-1.85765233e-03,2.77220245e-04,3.00309213e-03,1.52180246e+01
                    ,7.36706698e+00,1.75512239e+01,-2.63048025e-07,9.41282197e+00
                    ,-9.38418631e-03,6.43320965e-01,1.54049196e-02,-1.56825647e-05
                    ,-4.33653982e-02,2.02881588e-02,-6.27918877e-03,1.76681590e-01
                    ,5.55399099e-03,-2.15578845e+00,-1.61810849e+00,8.85559345e-05
                    ,-1.15546704e-05,-2.22619867e+00,-7.95237753e-03,2.23355942e-04
                    ,-1.89418041e+00,2.27201666e-02,-4.85903382e-03,1.70125365e-01
                    ,7.24564457e-03,-3.77284287e-06,2.19613366e-04,-1.02884772e+00
                    ,-7.94482891e-03,-3.46664894e-02,-1.68514497e-01,5.09850617e-03
                    ,4.97809002e-02,-1.73300721e-04,2.52911109e-01,1.29441772e-04
                    ,-1.35062748e-01,5.13748557e-06,1.99030605e-04,-4.59649326e-01
                    ,-7.59394318e-02,-5.45538536e-03,-2.36305825e+00,-3.04792990e-02
                    ,4.81545068e-01,1.52852381e-04,2.32386370e-02,-4.49262698e-04
                    ,-1.32472803e-01,-4.14558280e-06,2.78688024e-02,8.27103296e-06
                    ,-2.92694283e-03,-1.85914206e+00,3.03771049e-04,6.60339693e-02
                    ,4.71945708e-01,1.77730274e+01,6.02032289e+00,1.71419510e+01
                    ,4.01133521e+00,9.14432015e+00,1.90170856e+01,-1.81896952e-01
                    ,-1.36663800e-01,-5.78120240e-02,1.62152673e+00,-4.64818084e-02
                    ,-6.26805627e-01,4.40956274e-01,6.90149140e-01,-1.28740068e+00
                    ,1.19736368e-04,1.96961980e+00,1.17089682e-04,-1.21783186e+00
                    ,1.48161253e+00,1.46487064e+00,-7.12183541e-01,-7.70843431e-01
                    ,3.48453810e-01,8.72078650e-01,-4.08181851e-02,8.07673412e-02
                    ,-1.41482026e-01,-1.61202378e+00,2.09646308e-03,1.18397701e-05
                    ,9.05213569e-02,6.14348942e-04,-4.52315927e+00,9.63542041e-03
                    ,2.15673881e+01,-5.10021870e-03,5.24376883e-04,-3.59629840e-01
                    ,5.52359181e+00,1.65952773e+01,-1.52746468e-02,-1.11823712e-02
                    ,-8.38954593e-04,2.75303006e-03,7.29087153e-08,1.64969595e+01
                    ,5.12990039e-02,1.67064231e-01,1.84861872e-02,1.98996230e+01
                    ,1.97080737e+01,1.01816745e-04,5.30748095e+00,1.64297724e+01
                    ,1.25715810e+01,-1.38383242e-03,2.40920886e-02,1.26413132e-05
                    ,9.50950969e+00,-4.22283469e+00,5.85597916e-01,3.27768517e-01
                    ,-1.09013097e+01,-1.18719409e+01,-1.26694797e+01,1.41146167e+00
                    ,-6.60968351e+00,-1.78337661e+01,-1.42551144e+01])
            
                # Define the dependent variable
                name_dep_var = crosssell_types_max
                # Say which covariates we are going to use
                name_covariates = full_covariates
                # take a subset of the number of periods, just to test
                df_periods  = dflist[:5] # only use 5 periods for now?
                #Define number of segments
                n_segments = 5
                run_cross_sell = True

                          
            if (source == "finalActivity"):
                
                # note: dataset used is "final_df_finB04"
                param_cross = np.array([ 1.40317466e-01,  4.52759878e-01,  1.53728048e-04, -1.29455500e-01,
                -1.67580052e-01,  5.85788986e-01,  1.78204713e-01, -5.58905705e-04,
                -8.54111694e-01,  1.15437545e-01, -3.13876124e+01,  1.60574429e+01,
                1.55167454e+01,  1.56559190e+01,  1.56975497e+01,  1.56886264e+01,
                1.58385134e+01,  1.56247325e+01, -4.32139415e-03,  1.55515136e-01,
                9.14039679e+00, -1.58427129e+00, -7.66832113e+00,  7.09329805e+00,
                3.82200420e+00, -3.30208583e+00,  2.89148035e+00,  1.29348817e+00,
                7.77530885e-01,  9.78623946e-01, -1.21755124e-01, -2.62002794e-01,
                -1.13043252e-02, -9.79775511e-01,  2.16267069e-01,  1.37262204e+00,
                1.50048290e-01,  2.46021310e-02,  3.65928234e-02, -9.24598047e-02,
                -1.41507395e-01, -2.53221408e-01,  1.73864829e-01,  1.20410067e-01,
                6.86408411e-03, -3.52351529e-03,  1.21852569e+01,  1.13284305e+01,
                1.71098589e+01,  8.40967169e+00,  1.83085345e+01,  5.07705129e+00,
                7.18670047e+00,  1.21302640e+01,  4.38836635e+00,  4.76497637e+00,
                5.80298099e+00,  1.24577304e+01,  9.39271385e+00,  9.76444572e+00,
                6.51689729e+00, -1.54844265e+00, -5.38946312e+00, -2.56214536e+00,
                -9.65359438e+00])

                ## Define the dependent variable
                activity_dummies.extend(activity_total_dummies)
                name_dep_var = activity_dummies
                # Say which covariates we are going to use
                name_covariates = personal_variables2
                # take a subset of the number of periods, just to test
                df_periods  = dflist[:12] # use first 3 years
                #Define number of segments
                n_segments = 3
                run_cross_sell = False
            
            print(f"dependent variable: {name_dep_var}")
            print(f"covariates: {name_covariates}")
            print(f"number of periods: {len(df_periods)}")
            print(f"number of segments: {n_segments}")
            print(f"number of people: {len(dflist[0])}")
            
            
            hmm = ht.HMM_eff(df_periods, name_dep_var, name_covariates, 
                             covariates = True, iterprint = True)
            
        # Now interpret & visualise the parameters 
        p_js, P_s_given_Y_Z, gamma_0, gamma_sr_0, gamma_sk_t, transition_probs = hmm.interpret_parameters(param_cross, n_segments)
        
        if run_cross_sell == True: # do we want to run the model for cross sell or activity
            tresholds = [0.2, 0.7]
            order_active_high_to_low = [0,1,2]
            active_value_pd = pd.read_csv(f"{outdirec}/active_value.csv")
            active_value = active_value_pd.to_numpy()
            active_value = active_value[:,1]
            dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                      active_value, tresholds, 
                                                                                                      order_active_high_to_low)
        else:
            t = 5
            active_value  = hmm.active_value(param_cross, n_segments, t)
            active_value_df = pd.DataFrame(active_value) 

            active_value_df.to_csv(f"{outdirec}/active_value.csv")

        n_cross_sells = hmm.number_of_cross_sells(cross_sell_target, cross_sell_self, cross_sell_total)

    
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

        
        
        
        
        
        
        

