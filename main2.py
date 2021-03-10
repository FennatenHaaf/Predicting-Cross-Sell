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
            n_segments = 6
            reg = 0.1 # Regularization term
            max_method = 'Nelder-Mead'
            
            outname = f"activity_n{len(dflist[0])}_seg{n_segments}_per{len(df_periods)}"
        
        
        #---------------- Enter initial parameters ---------------
        
        initial_param = None
        initial_param = np.array(
            [4.80402899e+00, 1.91624333e+00, -3.88881503e-02, 4.28199785e-09, -1.21063883e-01, -2.43999323e-05, -2.90416680e-01,
             -3.42758949e-05, -1.41598279e-04, 4.66321039e-01, 8.09630384e-01, -1.17652970e-11, -2.22259448e-06, -9.48294445e-13,
             -1.76238577e-05, 6.51284987e-09, 4.93941076e-08, -3.44394833e-01, -1.53266055e-01, 5.61778665e-14, -9.34820379e-08,
             3.44979455e-09, -6.19193540e-06, 5.94868264e-10, 1.53714534e-08, 1.23702644e-03, 2.07970787e-09, -5.30269530e-21,
             2.62406352e+00, -3.36064325e-01, -8.15310167e-07, 8.57073155e-01, 8.89985457e-02, 1.06667508e-01, -1.00153989e-08,
             2.75722856e-01, 4.77322979e-04, 8.08146843e-13, 2.47868988e-01, 1.83042250e+00, 1.17002846e-06, -3.68192829e-01,
             4.26042659e-04, -6.87122881e-06, 3.18540511e+00, 2.31261526e+00, 7.14266800e-01, -7.12612347e-02, -5.72603703e-07,
             6.51437676e-03, 2.65997822e-08, 5.29300781e-08, 2.53084726e-03, -2.29774181e-02, -8.21907427e-06, 2.93178993e+00,
             4.08955266e-06, -3.56103638e-09, -1.52827589e-04, -2.97762441e-07, 1.49771603e-12, 2.95851200e-06, -1.44527218e-12,
             5.49144905e-03, -1.76069420e-10, 6.95382516e-09, 1.04813710e+00, 2.90742016e-11, -9.23972824e-07, 5.03926174e-06,
             7.67001467e-05, 1.99845300e-01, -2.02442198e-14, 5.73376565e-09, -1.08897700e-06, 7.49269491e-11, 8.29387093e-07,
             2.72310670e-06, -7.67401408e-05, 3.39510746e-03, -1.63806372e-06, 1.54632236e-07, -6.00256548e-07, 2.17797921e-06,
             -5.37168339e-09, 9.47239901e-10, 8.48926659e-06, -1.73984410e-02, 1.36466896e-06, 1.41549348e-13, 9.56932642e-05,
             -1.20899999e-05, 5.62639561e-06, -2.43242378e-11, -2.39575334e-06, 5.34423516e-14, -5.89070592e-09, 6.18078930e-11,
             3.12747012e-09, -8.35981123e-10, -5.71949932e-08, 4.82795858e-09, -5.06847107e-04, 4.59284517e-05, -3.51137386e-05,
             6.19539982e-03, 6.89473796e-15, 9.34473523e-06, 2.06041377e-08, 1.35673769e+00, -1.20640751e-03, 1.48710891e-05,
             9.29385710e-04, -2.87242465e-09, 4.45932626e-05, -9.40883721e-01, 9.05656052e-06, 5.24209753e-05, 3.94022168e-09,
             1.79798591e-06, -5.49024426e-10, -2.48804712e-08, -7.25029574e-06, -1.98052713e-09, -9.26264721e-10, -7.88789331e-16,
             4.70175380e-04, -5.22426090e-08, -2.52463552e-10, -1.22827319e-03, 1.72659342e+01, 1.80262143e+01, 1.34692245e+01,
             1.59301448e+01, 1.67391968e+01, 8.46081345e-07, 5.95099454e+00, -9.49810937e+00, 4.36285568e+00, -2.62154958e-08,
             7.46406810e-08, 5.00513738e+00, 1.27909603e-01, -1.05248126e+00, -3.74443611e-04, 3.34216610e+01, 3.63383256e-09,
             1.70856875e-07, 9.40080354e-01, 4.49330618e-11, -3.08471741e+00, 2.84962244e+01, -3.08949706e-17, -7.88572502e-07,
             -3.92278773e+00, -2.73554113e+01, -7.24748231e+00, -2.01166353e-10, 1.21967784e-07, 5.02112726e-06, 1.84820400e+00,
             -5.50515261e+00, 3.11588067e-03, 2.72444696e+01, 3.51006882e-06, -7.09088745e-04, 9.47859568e-01, 3.32862432e+00,
             3.40185088e-06, -1.44498204e-05, -2.71608957e+00, 6.05157031e-01, 5.61962138e+00, 3.37148791e+00, 7.65684861e-05,
             1.33734357e+00, -3.14127371e+00, -5.16700272e-08, 5.56476397e-01, -6.96409106e-01, 6.46273621e+00, -8.56874394e-08,
             4.43435924e+00, 3.68977529e+00, 2.90516979e+00, 1.72745509e+00, 1.68876717e-08, -1.93886728e+01, 1.21891937e-09,
             -1.31512648e-04, -7.24542577e-09, 1.03372557e+00, 1.69592158e-03, -5.67716278e-06, -5.29139349e-01, 1.03208580e-07,
             1.62748320e-08, 8.89784905e-01, -1.97885559e-07, -3.75436084e-03, -1.80467722e-07, -6.89692651e-01, -2.62093562e+00,
             -6.50704044e-01, -5.45593655e-01, 1.23992004e-01, 1.31472103e+00, -2.64728620e-11, 5.06885872e-14, 9.67960024e-01,
             -1.17277215e-06, 2.70507790e-05, -1.01266186e-05, -1.46742286e+01, -4.10904027e-08, -1.05073589e-06, 4.52824818e-07,
             9.69129113e-01, 5.56288814e-01, 2.78705613e-01, 1.89228905e-01, -2.30859032e-13, 1.74671603e+00, 8.70952860e-01,
             1.29230984e+00, -4.96369133e-12, -1.24322778e-02, -2.36122464e-01, -1.36578375e-03, 8.99528765e-07, -1.01991296e-09,
             2.22939863e-01, 7.04054974e-04, -1.43840214e+00, 1.91538798e-09, 5.99489472e-01, -2.88709782e-16, 1.99664073e-01,
             1.30371540e-09, -1.48874373e+01, -9.69960099e-04, 2.53137075e-04, 1.43274693e-06, -2.83002559e-09, 7.25469332e-01,
             -4.32802283e-09, 1.09908439e-09, 1.00866872e+01, 1.24726397e+01, 7.41179924e-01, 1.75503364e-02, -2.22450385e+00,
             -5.24318111e-10, -7.75855681e-06, 1.16477706e+01, -4.67818465e-07, 1.03745348e-10, 5.82670503e+00, 1.52097618e-12,
             -2.14748452e-10, 3.85990205e-01, -1.07091171e-02, -9.37595580e-13, 1.24437486e+00, -2.19576747e-01, -2.72514727e-15,
             4.45502080e-06, 8.03936969e-05, 9.51693518e-10, 1.46802225e+00, 1.12280912e-06, 1.04812520e-04, -3.13923572e-06,
             2.24592146e-04, 1.38900249e-04, -7.46909004e-02, 1.03362379e-06, 8.67327855e-08, 1.15093643e-07, -8.21852740e-06,
             -5.10354964e-05, -1.57867111e-06, 7.56066353e-06, 1.99644877e+00, -4.08875107e-07, -2.51987223e-09, 1.85174893e-08,
             2.24575977e-04, -1.54608831e+00, 5.59861727e-01, 1.49587285e+01, 1.80217658e-09, 1.20891248e+01, 1.37267186e+01,
             1.39533347e+01, -3.57159218e+00, 1.01783090e-01, 1.03540014e-07, 7.91539104e+00, -8.88293806e+00, -3.00393683e-04,
             -1.03688803e+01, -1.49367602e+00, -2.58032997e-05, -6.67600687e-02, 1.13973620e-11, -1.77937326e-07, 1.22271411e-03,
             -8.40721515e+00, -3.22251332e+00, 6.74766379e-13, -7.12803766e-03, 2.03421313e-13, 1.09117151e+01, -5.04229553e-04,
             -1.04556229e+01, 1.03178036e-07, -1.15258320e-09, -2.97061410e-08, 2.41871269e-11, -8.98845060e+00, -1.58971526e-04,
             -1.02092138e+01, 1.53912788e+01, 1.64028417e+01, 1.59904119e+01, -7.65897158e-01, -1.87748751e-11, 8.34974843e+00,
             -2.75998464e+00, -7.16188277e+00, -9.67761460e+00, -1.02817106e+01, 2.23280214e+00, -1.10571795e+00, -9.22573306e+00,
             6.64461360e-17])
        
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
                param_cross = np.array(
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

        
        
        
        
        
        
        

