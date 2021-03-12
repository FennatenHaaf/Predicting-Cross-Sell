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
import seaborn as sns
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
    run_cross_sell = False # do we want to run the model for cross sell or activity
    interpret = True #Do we want to interpret variables
    saldopredict = False # Do we want to run the methods for predicting saldo

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
        
        # Plot for three interesting variables
        DI.plotCategorical(df, "income", xlabel = "Income category",
                           colours = "Blues")
        DI.plotCategorical(df, "age_bins", xlabel = "Age",
                           colours = "Blues")
        DI.plotCategorical(df, "saldototaal_bins", xlabel = "Total account balance (EUR)",
                           colours = "Blues")
        
        #Plot cooccurrences of portfolio ownership
        DI.plotCoocc(df, crosssell_types_dummies)
        
        print("*****Visualising cross-sell difference data*****")
        # Now visualise cross sell events from the full dataset
        
        diffdata = additdata.create_crosssell_data(dflist, interdir) 
        increase_dummies = ["business_change_dummy", "retail_change_dummy",
                               "joint_change_dummy",
                               "accountoverlay_change_dummy"]
        change_variables = ["business_change",
                                "retail_change",
                                "joint_change",
                                "accountoverlay_change"]   
        
        daatacs = DI.plot_portfolio_changes(diffdata, change_variables,
                                            percent = True)
        daatacs = DI.plot_portfolio_changes_stacked(diffdata, change_variables,
                                            percent = True)
        daatacs = DI.plot_portfolio_changes(diffdata, increase_dummies,
                                            percent = False, legend = False,
                                            xlabel = "Portfolio type",
                                            colours = "Blues")
        
        # Plot how often they are purchased together (only looking at portfolio increases)
        DI.plotCoocc(diffdata, increase_dummies)

    
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
            n_segments = 5
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
        # initial_param = np.array([ 3.99872551e-01, 1.55174007e+00, 3.01236578e-01, 3.60653223e-01,
        #         2.44391119e-01, 1.97533254e-01, -1.21325482e-01, -3.75900089e-01, -1.09597705e+00, 1.69133466e-01, 
        #         -3.71668823e-01, 1.03317680e+00, -4.56861382e-01, -1.86047755e-01, -1.82113764e-01, 6.10210129e-01,
        #         5.35256621e-01, 4.37430282e-01, 1.46450243e-01, 1.46182996e-01, 8.81644633e+00, 1.38062593e-01, 
        #         -9.14871380e+00, 5.61629100e+00, 4.06053603e+00, -2.47751929e+00, 1.12557711e+00, 1.23822825e-01, 
        #         -7.39454264e-01, -6.66564174e-01, 1.24907110e-02, -5.47088303e-01, 5.10578630e-04, 5.36773181e-02,
        #         4.70270887e-01, -8.83672811e-04, -1.02892266e+00, -1.12122780e+00, -1.33884713e+00, 4.74891845e-01,
        #         3.23462965e-01, 2.61437448e-01, 1.10933134e+00, -9.80237698e-02, -1.57410914e-03, -8.42882097e-01, 
        #         8.86916524e+00, 1.08789822e+01, 1.54395082e+01, 9.31348666e+00, 1.86245690e+01, -4.32810937e-04, 
        #         2.08956735e+00, 7.48082274e+00, 3.96433823e+00, 5.74870230e+00, 6.32090251e+00, 8.66492623e+00, 
        #         8.10626061e+00, 8.41162052e+00, 4.28823154e+00, -1.71701599e+00, -5.80767319e+00, -3.19105463e+00, 
        #         -8.70848005e+00]) 

        
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
                                                             random_starting_points = False)  

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
             
            #source = "finalActivity"
            #source = "crosssell4seg"
            source = "crosssell5seg" 
            #source = "crosssell6seg"
            
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
                
                # Na BFGS stap - loglikelihood 10779.20266428308
                # param_cross = np.array([ 3.89261727e-01, 1.56034975e+00, 3.09612435e-01, 3.70011248e-01, 2.52680249e-01, 1.99835992e-01, -1.18796289e-01,
                #                         -3.73533007e-01, -1.09442494e+00, 1.69065293e-01, -3.77208139e-01, 1.04161737e+00, -4.48615818e-01, -1.77206832e-01, 
                #                         -1.73637174e-01, 6.07338913e-01, 5.32424562e-01, 4.34502792e-01, 1.42545230e-01, 1.46008940e-01, 8.77258469e+00, 1.15444484e-01, 
                #                         -9.16975488e+00, 5.58510178e+00, 4.05141335e+00, -2.48416290e+00, 1.13763980e+00, 1.47168271e-01, -7.15420332e-01, -6.42640392e-01,
                #                         8.75183945e-03, -5.49714992e-01, -2.52451947e-08, 4.63836545e-02, 4.71423728e-01, -3.63765157e-08, -1.01656520e+00, -1.10814438e+00,
                #                         -1.32566461e+00, 4.69633537e-01, 3.18529557e-01, 2.57103676e-01, 1.10201666e+00, -9.69667165e-02, 2.46845602e-09, -8.39344164e-01,
                #                         8.87035571e+00, 1.06290697e+01, 1.51811554e+01, 9.22819687e+00, 1.84922101e+01, -2.09568257e+00, -1.70267491e-09, 5.39054524e+00,
                #                         3.96414670e+00, 5.74004768e+00, 6.32191420e+00, 8.61859730e+00, 8.17545291e+00, 8.48063569e+00, 4.35777877e+00, -1.71680718e+00,
                #                         -5.79908406e+00, -3.19123265e+00, -8.66145072e+00]) 

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
                run_cross_sell = False
                outname = f"interpretparam_activity_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
                
                #df = pd.read_csv(f"{outdirec}/{outname}_standarderrors.csv")
                #arr = df.to_numpy()
                #param_cross = arr[:,1].astype('float64')
                
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
                outname = f"interpretparam_crosssell_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
                
                #df = pd.read_csv(f"{outdirec}/{outname}_standarderrors.csv")
                #arr = df.to_numpy()
                #param_cross = arr[:,1].astype('float64')
                
            if (source == "crosssell5seg"):
                # fenna results met loglikelihood 8502.154086118475
                # note: dataset used is "final_df_finB04"
                # param_cross = np.array([ 4.19867310e+00, 6.06015209e+00, 2.37976143e-01, -6.64865721e-01, -4.12053659e-05, -6.80088543e-02, -4.11376847e-03, 3.88857655e-01, 2.00450931e-05, 2.70608141e-07, -5.61241559e-01, 8.43083490e-05, 1.40596669e-06, -1.07061592e-08, 8.04423755e-02, 6.13768356e-02, -1.28354986e-04, -1.32369389e+00, -6.80543286e-07, 3.80317376e-03, -3.52459625e-02, -2.65383337e-02, -1.26437996e+01, -9.70072510e+00, 3.74489452e-05, -1.84209430e-08, -1.21529891e-10, -1.63393508e+00, 6.31001746e+00, 1.85973035e-01, -4.60900509e-01, 7.23505477e-01,
                #                         -5.79910905e-11, -2.78709427e-05, -1.92956881e+00, -2.01952351e+00, -3.50593768e+00, -3.30149668e+00, 8.02344788e-07, 1.37244256e+00, -4.54699358e-15, 7.18617469e-01, -1.16574293e+00, -4.76513301e-05, -4.58481882e+00, -9.43100317e-01, 3.30139559e-01, -1.46519473e+00, 3.07170955e-07, 3.51960406e+00, 1.88931152e+00, 1.28357709e+01, 1.59580860e+01, 1.65564415e+01, 7.96072920e+00, 5.36006759e+00, 1.18622758e-02, 3.59979256e-05, 5.13486582e-01, 2.83701522e-01, 1.01562202e-01, -4.17664495e-02, 1.54737505e-01, 2.04681547e-01,
                #                         -5.36179216e-17, 4.17843227e-01, -3.68866810e-01, -1.51643262e-01, -4.71683549e-03, -2.44488990e-04, -1.07947955e-02, -4.21503291e-01, 6.60551326e-01, 3.37311599e-01, 2.19870127e-01, 3.30004375e-01, -1.28057777e+01, -8.80375263e+00, 3.47121964e-05, 2.68272331e-05, 6.21237363e-06, -3.59373459e-01, 4.68925560e+00, -1.01379959e+00, -2.90753957e-06, -2.02851290e+00, 2.37750017e-03, -3.47973644e-01, 2.27099315e-01, 1.83661799e+00, 3.92795598e+00, 3.00608272e+00, -1.90014319e-03, -4.83057482e+00, 2.12331386e-13, -2.36358215e+00,
                #                         1.70372776e-06, 1.38746349e-03, -1.70887363e-08, 2.14184329e-01, -2.81959270e-03, 7.35118180e-01, -1.13808522e-06, 7.05625839e+00, 1.92285717e+01, -5.41823924e-05, -2.50050767e-05, -2.46973589e-07, 4.80097531e+00, -3.31949076e+00, -6.49927513e-04, -5.44693344e+00, -4.97340890e-03, 7.01561306e-07, -1.51167574e+00, 5.27260920e-04, -4.21267979e+00, 1.67109460e-03, 
                #                         5.15070350e+00, 2.47285507e+00, 1.01765190e+01, 1.48357221e-07, 3.15744426e-03, 6.95551206e-06, 4.41770872e-09, -3.76411655e-08, 1.44150438e-12, -1.68356833e-04, 6.20942979e+00, -2.54743875e-04, 6.20261687e-01, 2.96150320e+00, 1.84176795e+00, 2.56780672e+00, 1.12594573e+00, 8.83447047e-01, 1.14236507e+00, 2.47078658e-07, -4.45529788e-03, -4.47564065e-01, 3.70626452e-08, 2.37388164e-04, -3.90460792e-01, 4.09616521e-05, -1.18740436e+00, -1.10752691e+00, -1.76559471e-01, 2.04921820e-06, -7.49737179e-03, -1.04269540e-02, 
                #                         3.61952370e-08, -9.60019557e+00, -1.25433200e+01, -3.16697240e-03, 6.04063434e+00, -2.36234755e-01, -1.27128770e-07, -5.91423830e-01, -3.47314386e-08, -7.77108894e-04, -1.83758536e-02, -1.89584563e-01, 1.76745863e-03, 1.08429244e-07, -1.18947761e-02, -5.49556582e-04, 4.03984868e-06, -7.98963107e-07, 8.66756613e-07, 2.71381004e-05, -3.89614418e-06, -4.67892884e-07, 3.11030459e-03, 4.80101427e-06, -3.53152511e-03, 1.50484916e+01, 1.46637387e+01, 1.37479574e-09, -8.07259516e-04, 1.28559140e+01, 5.91512085e+00, -2.00813625e-01, 
                #                         9.38313418e-05, -1.55120343e-01, -8.91438504e-05, 3.12055261e-01, 2.85778729e-07, -1.44033727e-07, -1.75668296e-05, -1.61330781e-05, -1.32107096e-02, -1.49776879e-04, -3.59865093e-04, 2.68147489e-03, 1.54207356e-04, 8.32957884e-01, -1.87100044e-08, 2.36860609e-02, 1.18913391e-06, -1.93672636e-01, -1.86530104e-02, -9.98431150e-11, -1.10390211e+00, -1.30094217e+01, -1.15755666e+01, -4.32047055e-05, 5.33452413e-04, 1.39561332e-11, -1.29667900e-02, 7.41593409e-01, 2.45370857e-03, 2.00631345e-04, 2.32166274e-06, -2.91257026e-07, 
                #                         -1.71838677e-04, 1.02135909e-07, -2.65456311e-03, 1.00946766e-07, 3.67319529e-04, -8.96608429e-07, 6.10268937e-06, -8.08933588e-10, 6.23259601e-01, -1.72753249e-05, -1.57391102e-02, -2.05761239e-07, 4.38671335e-08, 5.25111097e-04, 1.25285441e+01, 1.48192241e-06, 7.19880991e+00, -1.41556688e-04, -2.51362919e+00, -7.95536688e+00, -1.25125228e+01, 1.06597625e-02,
                #                         -2.48728797e-08, -1.04578075e+01, -7.30495286e+00, 1.68686343e+01, 1.00449793e+01, 4.47533368e+00, 6.57446264e-02, -9.34506970e+00, -8.59056760e-01, 4.74698051e-01, -3.19128972e+00, -8.71488273e+00, -1.31997583e+01, 9.73555967e+00, -1.92787375e+01, -1.11493867e+01, -8.01254111e+00, 1.88225717e+01, -9.50835965e-10, -8.63422627e-04, -2.33724213e-01, -9.46339233e+00, -9.95915257e-01, -5.25865714e+00, -7.93216746e+00, -2.47030362e+00, 2.07958549e+00, 1.43403669e+00, 8.35978601e+00, 3.63464444e-14, -3.11424080e+00]) 

                # This is from after 1 more BFGS step!! Loglikelihood: 7138.523375256655
                param_cross = np.array([ 4.13648592e+00, 6.98877960e-01, 3.54922212e-01, -7.03593918e-05, -1.69171597e-05, -8.09983972e-05, -4.38343059e-06, 4.31253355e-01, 1.16508587e-06, 5.19458902e-08, -3.36080590e-03, -2.69735931e-05, 3.49702373e-01, 1.54262831e-01, 8.81092414e-02, 6.50931238e-02, 6.79524274e-02, -7.61990694e-01, -8.08017164e-05, -4.36324941e-05, -2.99162916e-06, -1.25460478e-08, -4.09332816e+00, -5.17444554e+00, -2.11602956e-08, -6.98751623e-07, 3.17577260e-06, -9.03837502e-01, 5.34055494e-01, 3.46528415e-01, -4.51016028e-02, 4.59786755e-01, 1.13096603e-01, 9.42966425e-06, -1.09166220e+00, -1.28938054e+00, -1.07724441e+00, -1.32504119e+00, -5.04824228e-01, 3.79398900e-01, -4.12799588e-01, 6.59207166e-02, 2.95175621e-07, -6.88009378e-01, -8.00571833e-01, 9.50047057e-02, 6.02029316e-01, 4.96291853e-02, 8.85754176e-01, 9.84824605e+00, 5.66001428e+00, 1.22371488e+01, 1.38161426e+01, 1.44749873e+01, 7.90486679e+00, -2.14307079e-06, 1.29181448e-01, 6.69284902e-01, 5.19213970e-01, 3.57244454e-01, 1.12064993e-01, -3.87730446e-07, 1.55998858e-01, 2.04388973e-01, 5.61242160e-01, 4.15530842e-01, -7.26747795e-04, -5.99774479e-06, -1.13644748e-06, -8.53365373e-06, -3.27436224e-05, 1.15477569e-01, 6.44196162e-01, 3.16005785e-01, 2.38116623e-01, 3.37443756e-01, -3.96993270e+00, -4.66451422e+00, -6.95404617e-06, 2.18928386e-06, -4.97553646e-07, -3.15267571e-08, 1.00709927e-01, -2.43614766e-05, 1.87998644e-05, -8.68883080e-01, -4.56157289e-01, -2.85296372e-02, 2.45496246e-06, 6.39677858e-01, 5.02180512e-01, 5.47586106e-01, 9.25721856e-02, -3.75798346e-02, 8.37704282e-01, -1.35712375e-01, 3.00465538e-07, 1.63859973e-04, 8.69043826e-04, -4.76375333e-01, -1.52322817e+00, -7.41578353e-01, -5.98310234e-01, 1.22732821e+01, 1.34865512e+01, 2.06314833e-07, -5.01350406e-07, -5.83043310e-06, 4.58045736e+00, -1.56926369e+00, -2.09045205e-06, -9.76458075e-09, -1.24758787e-07, -3.33904140e+00, 8.60348615e+00, 1.00807110e-01, -2.08127684e+00, -3.94439109e-06, 5.00720823e+00, 7.52396118e+00, 1.02440301e+01, 5.77287526e+00, 2.54397121e-06, -3.85225549e-07, -7.37149966e-01, -2.23325710e+00, 1.69341595e+00, -1.62258570e-06, 4.03863661e-01, 4.62957503e-01, 6.66567024e-01, 3.09395474e+00, 1.85114254e+00, 2.25204232e+00, 1.15939979e+00, 9.50967162e-01, 1.19563752e+00, 1.52422724e-07, 3.27115833e-02, -5.14735123e-04, 1.15723675e-02, 3.08722216e-02, -2.58087871e-06, -7.08287776e-06, -8.76746830e-04, -3.87891188e-06, -5.06791288e-05, 1.94948271e-01, -6.40855545e-07, -2.66007782e+00, -7.45111089e-01, -5.12451612e-01, -1.44119563e+00, 1.91182068e-06, 1.19224643e+00, -1.36558362e-01, -3.26195149e-06, 3.64140704e-01, -6.75968980e-01, 1.45921440e-09, -3.92810832e-01, -1.37310253e+00, -2.85087226e-01, -4.53003716e-08, -1.03984075e+00, 3.89602751e+00, 2.10421073e+00, 5.68049419e-01, -1.02943458e+00, -1.01967610e+01, -3.51405957e+00, -3.57473917e+00, -4.15434119e+00, -3.18158371e+00, -3.41475867e+00, 7.90046309e+00, 5.77008353e+00, 3.17167240e-01, -1.83995534e-10, 1.39759470e+01, 1.12961626e-01, 2.59385313e-01, -2.84082301e-05, -6.20949009e-06, -3.69472119e-06, 3.06262954e-07, 5.43632976e-06, 2.91267780e-07, -1.80554599e-06, -2.95164619e-01, 2.79414082e-06, 4.36400719e-01, -1.87482037e-05, -4.84042355e-05, 3.89353711e-01, 8.35882054e-01, 1.18247646e+00, 1.12717645e+00, 1.44403764e-01, 1.47534464e-07, 2.04014517e-06, -5.51273585e+00, -3.45283562e+00, -4.52617906e+00, -3.51627929e+00, 1.27952130e-06, -1.22004864e-05, 3.12104009e-01, 3.17670786e-06, -1.41471219e+00, 1.88629952e-06, -1.11773555e+00, 1.24607863e-05, 1.83579887e-06, 2.37874748e-01, 3.52945308e-06, 1.75369746e-08, -1.14255646e+00, 1.45670299e-06, -5.02306604e-01, 3.26478583e-06, 2.40584923e-03, 1.46299784e+00, 2.53835123e+00, 6.09840193e-01, 1.07942123e+00, 1.66453104e+00, 1.13381017e+01, 1.52930895e+01, 9.68663652e+00, 1.14831116e+01, -1.82298036e-06, -1.04384031e+01, -5.30887308e+00, -4.42756319e+00, -3.44050556e-07, 1.10080501e+01, -9.20694886e+00, 5.92768751e-07, 5.46602484e+00, 1.32260335e+01, 1.30857610e+01, 2.02062688e-01, -2.37240014e-01, 1.56306573e-06, 9.77761698e+00, -1.11430446e+01, -6.01418925e+00, -5.13237391e+00, 9.68489089e+00, -9.98931861e+00, -9.90327267e+00, -6.86025163e-02, 1.06867796e+01, 4.73030038e-06, 9.70117869e-07, -3.31420414e-02, -2.27382000e-05, 4.51225618e-01, -7.38473799e-09, -9.71312390e-08, -5.13142882e+00, -6.01006055e+00, 1.44530372e+00, -9.41694105e-01, -1.23700647e+00, -1.10705419e+01]) 
                

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
                outname = f"interpretparam_crosssell_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
                
                #df = pd.read_csv(f"{outdirec}/{outname}_standarderrors.csv")
                #arr = df.to_numpy()
                #param_cross = arr[:,1].astype('float64')
                
            if (source == "crosssell6seg"):
                # note: dataset used is "final_df_finB04"
                param_cross = np.array(
                    [4.56256192e+00, 2.02826277e+00, -1.82638536e-01, 9.83834937e-10, 1.01887049e-05, 4.35316927e-07, -2.44616476e-01,
                     -3.29834706e-07, 4.63711897e-03, 1.90067247e+00, 7.28407671e-01, -7.24178529e-17, 3.51201776e-08, -1.99922212e-12,
                     1.79805080e-06, -2.18897373e-11, 2.68537680e-08, 6.56236707e-01, 3.35816521e-01, 1.95418714e-14, -4.97267288e-10,
                     -1.29761845e-10, -9.93933115e-05, 2.88894883e-10, -1.92959170e-08, -9.66551036e-03, -3.34019927e-11,
                     -3.42082282e-21, 5.36120102e-03, -8.52256741e-01, -3.72837432e-09, -4.50810345e-04, -4.08014223e-05,
                     -2.49805574e-04, -4.73064098e-13, 3.29076818e-03, -4.21550439e-05, 1.12961472e-12, -6.73229144e-01,
                     -1.79489811e-03, 1.44272406e-08, -6.83661582e-01, -1.69842437e-04, -1.33773340e-06, 5.03810163e-07,
                     9.23226358e-03, -2.49996074e-03, -4.32011132e-02, -5.55420151e-08, -3.65406486e-03, 1.89324922e-10,
                     -2.29803253e-09, 1.45807794e-02, 1.14294124e-02, -4.38138095e-08, 2.81965640e+00, 6.09702801e-06, 8.47637157e-11,
                     -1.29420681e-05, -1.89949808e-07, 7.64571071e-13, 4.04327625e-07, 1.02373085e-14, 1.56582782e+00, -6.03538767e-13,
                     1.00496030e-10, 3.25872618e-01, -3.51944541e-14, -1.57318380e-08, 5.99605018e-07, -3.75375075e-06,
                     -3.28641600e-03, 2.10129911e-15, -1.11740905e-08, -3.17981223e-08, -1.42955262e-10, -5.78675825e-08,
                     -7.65539006e-09, -2.93967010e-09, 3.43305047e-03, 2.06613579e-06, 1.84907669e-08, 2.47275141e-06, 1.82740515e-04,
                     -1.26647978e-12, -1.35203814e-13, 2.66205788e-06, 3.66076920e-05, 1.62350760e-09, 6.77929155e-16, -3.76548889e-04,
                     6.98420541e-08, -1.26355008e-06, -8.78769148e-13, -1.80624995e-04, -4.41351540e-17, -8.02326914e-10,
                     1.97088171e-13, -1.98691274e-11, -1.74344150e-09, 2.71553316e-10, 2.00472609e-11, -1.31808394e-03, 8.67627254e-10,
                     8.72135940e-06, 2.07645721e-06, -1.79569369e-16, 1.34242918e-03, 4.45824941e-09, 1.27774319e+00, 1.14084019e-03,
                     1.69078319e-09, 5.94423608e-03, -1.41386555e-11, -1.58125103e-07, -1.42983020e+00, 1.21029825e-06, 8.04774960e-05,
                     -1.62271592e-09, -3.64640599e-09, 3.08876435e-13, 4.41527337e-11, -1.51230427e-06, 4.43082682e-11, 5.67135223e-08,
                     -1.22548310e-17, 4.84212251e-04, -9.20807306e-08, 4.87358726e-12, -9.84649443e-04, 1.77819800e+01, 1.83576161e+01,
                     1.36206236e+01, 1.64098061e+01, 1.71278119e+01, 5.93972824e-07, 6.01090751e+00, -1.24493686e+01, 6.86562592e+00,
                     -8.76830014e-12, -5.81561275e-09, 2.05924079e+00, 1.45905710e-03, 1.86670874e+00, -7.30639061e-07, 2.13554866e+01,
                     1.20022105e-09, 2.62887508e-08, -2.37457011e-02, -5.18588853e-12, 1.92938709e-03, 7.63724849e+00, 1.25236646e-18,
                     -1.13094348e-05, -2.11446867e+00, -3.27255809e+00, -2.32700259e-01, 1.78190953e-12, 1.57615494e-11,
                     1.90129374e-05, 4.16082300e-04, -3.50102550e+00, -7.18019748e-03, 1.34729584e+01, 2.67037506e-06, 1.33496944e-04,
                     1.62258633e+00, -3.08000258e+00, -2.07118109e-09, 6.42245541e-06, 5.12260656e+00, 1.06060353e+01, 8.97490725e+00,
                     3.04419842e+00, -3.24771026e-08, 1.67036141e+00, -6.59999499e-04, -8.90718434e-11, 5.11531100e-01,
                     -7.98524461e-03, 4.31908643e+00, -1.13092465e-07, 3.74469668e+00, 2.28259041e+00, 2.48627119e+00, 2.32239353e+00,
                     1.08070962e-10, -2.90896680e+01, 1.95006207e-09, -1.52510790e-04, -6.07166795e-11, 8.17504695e-02,
                     -4.67740159e-05, 1.67196260e-07, 9.36306207e-01, -3.77330385e-12, 8.40029427e-12, 1.32730480e+00, -7.94390403e-10,
                     -1.82389237e+00, -3.43431811e-06, -3.41535216e+00, -1.93443972e+00, -7.82451576e-04, -2.01623305e+00,
                     1.07058586e-02, -2.97166270e-02, 7.79705638e-11, 8.97321022e-14, 2.82025756e+00, 9.17699092e-06, 2.02342494e-04,
                     -4.47064442e-11, -1.58406284e+01, -3.16792708e-07, 1.19759912e-07, 1.44848150e-09, 3.69688415e-01, 1.46706753e+00,
                     -6.09558848e+00, -3.69919404e-01, 1.77204534e-15, 3.37396320e+00, 7.12603024e+00, 5.13412693e+00, 1.42985838e-13,
                     -1.35860648e+00, 9.86773080e-01, 7.16527647e-05, -4.53020988e-07, 1.22554574e-14, -9.79684856e-01,
                     -2.48367558e-03, -4.08610451e+00, 1.68959329e-11, 1.57695419e-04, -2.81665209e-16, 9.54317319e-01, 1.13779464e-10,
                     -2.14200285e+01, -5.69815115e-03, 6.99977582e-05, 1.27401377e-06, -1.36657633e-12, -8.62183368e-01,
                     -1.17448900e-07, -6.84912908e-10, 4.76940567e+00, 1.22672644e-01, 6.35854594e-03, 2.06395952e-04, -1.71745973e+00,
                     1.18690533e-14, -2.64577059e-06, 3.74685204e-05, -9.35110500e-10, 1.27495106e-10, 4.72414520e-01, 5.68268584e-14,
                     -1.59403298e-12, -2.64751845e-02, 4.70743297e-06, -3.83230249e-15, -1.24884654e-02, -9.25683765e-05,
                     3.74877362e-19, -4.31021265e-07, 1.03898415e-06, 6.69438818e-11, 6.65972989e-01, -2.63645988e-07, -1.61711013e-04,
                     3.29563544e-09, 2.01551777e-07, -2.72712040e-04, 3.32406059e+00, 1.01609718e-09, -8.85495001e-09, -1.81529237e-08,
                     3.34898668e-08, 2.66923789e-07, 1.66921839e-10, 3.87767514e-05, 7.01322175e-02, 9.67579103e-09, -4.31915176e-14,
                     -6.13970431e-09, 3.46435646e-07, -9.22873474e-01, -5.15910912e-03, 2.37124847e+01, -1.18190506e-09,
                     1.48082163e+01, 1.81607555e+01, 1.82420777e+01, -4.03773274e+00, 9.86908036e-06, 8.86778581e-12, 9.67164039e+00,
                     -7.15417107e+00, -8.80073378e-06, -1.11456528e+01, -4.69599081e-04, 5.63508098e-07, -1.97338380e-03,
                     2.34615152e-11, 2.86579844e-09, 6.02457416e-07, -6.24327168e+00, -3.34420303e+00, 3.15807412e-14, -3.01765533e-03,
                     1.00394881e-17, 6.67942317e+00, 3.35536950e-05, -1.04539378e+01, -8.96312423e-10, -1.41261986e-13, 7.04753770e-09,
                     -1.47665955e-12, -4.05942022e+00, -2.12622319e-03, -5.07641447e+00, 1.55284372e+01, 1.67424584e+01,
                     1.63111928e+01, -8.10483751e-03, 5.77942174e-13, 8.62280815e+00, -2.75603952e+00, -7.08573681e+00,
                     -9.80490722e+00, -1.03927019e+01, 1.47401422e+00, -1.00801679e+00, -9.49654484e+00, -8.54052613e-18])
                
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
                run_cross_sell = True    
                outname = f"interpretparam_crosssell_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
                
                #df = pd.read_csv(f"{outdirec}/{outname}_standarderrors.csv")
                #arr = df.to_numpy()
                #param_cross = arr[:,1].astype('float64')

            print(f"dependent variable: {name_dep_var}")
            print(f"covariates: {name_covariates}")
            print(f"number of periods: {len(df_periods)}")
            print(f"number of segments: {n_segments}")
            print(f"number of people: {len(dflist[0])}")
            
            
            hmm = ht.HMM_eff(outdirec, outname,
                         df_periods, reg, max_method,
                         name_dep_var, 
                         name_covariates, covariates = True,
                         iterprint = True,
                         initparam = param_cross,
                             visualize_data = True)

            
        #----------------------------------------------------------------------------
            
        # Now interpret & visualise the parameters 
        #p_js, P_s_given_Y_Z, gamma_0, gamma_sr_0, gamma_sk_t, transition_probs = hmm.interpret_parameters(param_cross, n_segments)

        calculate_se= False   
        if calculate_se: 
            print("-----getting standard errors-----")
            hess_inv, dfSE, param_afterBFGS = hmm.get_standard_errors(param_cross, n_segments)
            print(f"Done calculating standard errors at {utils.get_time()}")
             
        # Get the targeting decision

        if run_cross_sell == True: # do we want to run the model for cross sell or activity
            print("-----Calculating targeting decision-----")
            tresholds = [0.2, 0.7]
            order_active_high_to_low = [0,1,2]
            active_value_pd = pd.read_csv(f"{outdirec}/active_value.csv")
            active_value = active_value_pd.to_numpy()
            dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                      active_value, tresholds=tresholds, 
                                                                                                      order_active_high_to_low = order_active_high_to_low)
            n_cross_sells = hmm.number_of_cross_sells(cross_sell_target, cross_sell_self, cross_sell_total)
            P_s_given_Z_hypo, P_s_given_r_hypo, Z_hypo = hmm.hypo_customers(False, param_cross, n_segments, interdir)
            
        else:
            print("-----Calculating active value-----")
            active_value  = hmm.active_value(param_cross, n_segments, len(df_periods))
            active_value_df = pd.DataFrame(active_value) 
            
            utils.save_df_to_csv(active_value_df, outdirec, "active_value", 
                            add_time = False )


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
          prod_own_new = testing_period[crosssell_types_dummies]  
          # These describe product ownership yes/no in the previous period
          prod_own_old = last_period[crosssell_types_dummies]
          
          ginivec = hmm.calculate_gini(prod_ownership_new = prod_own_new,
                                       prod_ownership_old = prod_own_old, 
                                       product_probs= prod_own, binary = True)
          print("Ginicoefficient for the binary ownership dummies")
          print(ginivec)
          
          # Now do it with the actual numbers of ownership
          prod_own_new = testing_period[crosssell_types_max]
          prod_own_old = last_period[crosssell_types_max]
          ginivec2 = hmm.calculate_gini(prod_ownership_new = prod_own_new,
                                        prod_ownership_old = prod_own_old, 
                                        product_probs= prod_own, binary = False)
          print("Ginicoefficient for the (non-binary) ownership variables")
          print(ginivec2)
          
          
          #--------------------- TP/ NP calculation -------------------------
          
          diffdata = additdata.get_difference_data(testing_period, last_period,
                                           select_variables = None,
                                           dummy_variables = None,
                                           select_no_decrease = False,
                                           globalmin = None)
            
          # These dummies describe whether an increase took place
          diffdummies = diffdata[["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]]  
    
          
          evaluation = hmm.calculate_accuracy(cross_sell_pred = cross_sell_self,
                                              cross_sell_true = diffdummies, 
                                              print_out = True)
         
# =============================================================================
# Evaluate thresholds
# =============================================================================

        evaluate_thresholds = False
        if (run_cross_sell & evaluate_thresholds):
            print("-----Plotting results for different thresholds-----")
            lower = [0.01,0.02,0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55]
            upper = [0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.96,0.97,0.98,0.99]
            order_active_high_to_low = [0,1,2]
            active_value_pd = pd.read_csv(f"{outdirec}/active_value.csv")
            active_value = active_value_pd.to_numpy()
            
            t = len(df_periods)
            last_period = df_periods[t-1]
            testing_period = dflist[10] # period 10 or 11 can be used
            
            def evaluate_threshold_plot(active_value, order_active_high_to_low ,
                                        testing_period, last_period,
                                        lower,upper,vary_lower=True,vary_upper=True,
                                        lower_base=0.2,upper_base=0.7):
                
                diffdata = additdata.get_difference_data(testing_period, last_period,
                                           select_variables = None,
                                           dummy_variables = None,
                                           select_no_decrease = False,
                                           globalmin = None)         
                diffdummies = diffdata[["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]]  
                
                if vary_lower:
                    low_bounds = lower
                else:
                    low_bounds = np.repeat(lower_base,len(lower))
                if vary_upper: 
                    up_bounds = upper
                else:
                    up_bounds = np.repeat(upper_base,len(upper))
                                
                sensitivity = pd.DataFrame()
                
                for i in range(0,len(low_bounds)) :
                    thresholds = [low_bounds[i],up_bounds[i]]
                    dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                          active_value, tresholds=thresholds, 
                                                                                                          order_active_high_to_low = order_active_high_to_low)                    
                    # Get accuracy measures
                    evaluation = hmm.calculate_accuracy(cross_sell_pred = cross_sell_self,
                                              cross_sell_true = diffdummies, 
                                              print_out = False)
                    
                    select = (evaluation["measure"]=="sensitivity")
                    sens = evaluation.loc[select,["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]]
                    sensitivity = pd.concat([sensitivity, sens], axis=1)
       
                    select = (evaluation["measure"]=="accuracy")
                    acc = evaluation.loc[select,["business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]]
                    accuracy = pd.concat([sensitivity, acc], axis=1)
                  
                sensitivity.columns = ["business","retail","joint","accountoverlay"]
                sensitivity["threshold_low"] = low_bounds
                sensitivity["threshold_high"] = up_bounds
                
                accuracy.columns = ["business","retail","joint","accountoverlay"]
                accuracy["threshold_low"] = low_bounds
                accuracy["threshold_high"] = up_bounds
                
                return accuracy, sensitivity
                             
            acc, sens = evaluate_threshold_plot(active_value, order_active_high_to_low ,
                                        testing_period, last_period,
                                        lower,upper,vary_lower=False,vary_upper=True,
                                        lower_base=0.2,upper_base=0.7)
            
            print(acc)
            print(sens)
            
            
# =============================================================================
# SALDO PREDICTION
# =============================================================================

        if (saldopredict & run_cross_sell):
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
           
            #--------- MAKE DATASET CONTAINING SALDO DIFFERENCES TO TRAIN ---------
            saldo_name = "saldopredict"
            predictdata = additdata.create_saldo_data(saldodflist, interdir,
                                            filename= saldo_name,
                                            select_variables = selection,
                                            dummy_variables = dummies,
                                            globalmin = globalmin)
            
            
            #--------- MAKE SALDO PREDICTION VARIABLES IN THE DATASET --------- 
            for i, df in enumerate(dflist):   
                dum, dumnames =  additdata.make_dummies(df,
                                                     dummies,
                                                     drop_first = False)
                df[dumnames] = dum[dumnames]
            
            
            #------------------------ TRAIN THE MODEL ------------------------ 
            print(f"****Create saldo prediction model at {utils.get_time()}****")
                    
            # The input dataset is the specific finergy dataset - if we want it
            # to be the full dataset we need to create dummy variables for that
            # dataset as well
            
            # Train the model
            predict_saldo = ps.predict_saldo(saldo_data = predictdata,
                                             df_time_series = dflist,
                                             interdir = interdir,
                                             )
            #We predict for period 10
            extra_saldo,  X_var_final, ols_final = predict_saldo.get_extra_saldo(cross_sell_yes_no = cross_sell_total, 
                                                                                 time=10, 
                                                                                 minimum = globalmin,
                                                                                 fin_segment = None)
            
            


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
        
        
        
        

