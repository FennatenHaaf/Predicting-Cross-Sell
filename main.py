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
    
    run_hmm = True
    run_cross_sell = True # do we want to run the model for cross sell or activity
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
        #initial_param = 
        
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
                n_segments = 3
                reg = 0.1 # Regularization term
                max_method = 'Nelder-Mead'
                run_cross_sell = False
                
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
        dfSE = hmm.get_standard_errors(param_cross, n_segments)
        
        if run_cross_sell == True: # do we want to run the model for cross sell or activity
            tresholds = [0.2, 0.7]
            order_active_high_to_low = [0,1,2]
            t = 9 
            active_value_pd = pd.read_csv(f"{outdirec}/active_value_t{t}.csv")
            active_value = active_value_pd.to_numpy()
            active_value = active_value[:,1]
            dif_exp_own, cross_sell_target, cross_sell_self, cross_sell_total, prod_own = hmm.cross_sell_yes_no(param_cross, n_segments,
                                                                                                      active_value, tresholds, 
                                                                                                      order_active_high_to_low)
            n_cross_sells = hmm.number_of_cross_sells(cross_sell_target, cross_sell_self, cross_sell_total)
             
        else:
            t = 9 # de laatste periode die als input in hmm is gestopt
            active_value  = hmm.active_value(param_cross, n_segments, t)
            active_value_df = pd.DataFrame(active_value) 

            active_value_df.to_csv(f"{outdirec}/active_value_t{t}.csv")


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
                globalmin = min(globalmin,saldo)
            
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

        
        
        
        
        
        
        

