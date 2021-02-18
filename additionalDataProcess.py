"""
Contains extra functions to transform data

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from datetime import date
import utils


# =============================================================================
# Methods for making saldo prediction dataset
# =============================================================================

def create_saldo_data(dflist, interdir, filename ="saldodiff", 
                      select_variables = None, dummy_variables = None ):
    """Make saldo data out of a df list"""
    i=0
    for df in dflist:
        if i==0:    
            # We assume our input is already transformed & aggregated!
            #dfold = aggregate_portfolio_types(df)
            dfold = df
        else:
            #dfnew = aggregate_portfolio_types(df)
            dfnew = df
            data = get_difference_data(dfnew,dfold,
                                       select_variables = select_variables,
                                       dummy_variables = dummy_variables)
            if i==1:
                diffdata = data
            else:
                # newcolumns = diffdata.columns.difference(data.columns)
                # for col in newcolumns:
                #     data[col]=0   
                # -> not necessary, if there is a difference in columns it will
                # just become nan
                
                diffdata = pd.concat([diffdata,data], ignore_index= True) #
                diffdata = diffdata.fillna(0)   
                
            dfold = dfnew
        i+=1
     
    diffdata["log_saldoprev"] = getlogs(diffdata["saldo_prev"])
    print(f"Done creating saldo prediction dataset at {utils.get_time()}" \
          f", saving to {filename}")
    utils.save_df_to_csv(diffdata, interdir, filename, add_time = False )
    return diffdata



def get_difference_data(this_period,prev_period, log =True,
                        select_variables=None, dummy_variables =None):
    """Get the datapoints for which there is a difference of 1 or more
    portfolios"""
    #TODO also add account overlay somehow so that we can also find the impact
    # of that (if there is any??)
    
    # Get the total number of portfolios in each period (excluding overlay)
    this_period['portfoliototaal']= this_period[["business","retail",
                                        "joint"]].sum(axis=1)
    prev_period['portfoliototaal']= prev_period[["business","retail",
                                        "joint"]].sum(axis=1)
    
    this_period['portfolio_change'] = this_period['portfoliototaal'] - \
                                      prev_period['portfoliototaal']
                                      
    saldo_prev = prev_period['saldototaal']
    saldo_now = this_period['saldototaal']
    minoverall = min(saldo_prev.min(),saldo_now.min())
   
    if log:
        # We take the log difference
        #saldo_prev = getlogs(prev_period['saldototaal'])
        #saldo_now = getlogs(this_period['saldototaal'])
        
        # Need to subtract the same number of each to take the log
        # and then get the log difference    
        print(f"Taking log difference, min of both is {minoverall}")
        logprev = np.log(saldo_prev+1-minoverall)
        lognow = np.log(saldo_now+1-minoverall)
    
        this_period['percdiff'] =lognow-logprev

    else:
        # Take the percentage - we add the minimum of the previous period to
        # deal with negative or zero values 
        print("Taking percentage difference, min of both is {minoverall}")
        saldo_prev2 = this_period['saldototaal']+1+minoverall
        saldo_now2 = prev_period['saldototaal']+1+minoverall
        this_period['percdiff'] =((saldo_now2-saldo_prev2) / saldo_prev2)*100
   
    # Get portfolio variables
    for name in (["business","retail","joint"]):
        this_period[name] = this_period[name].fillna(0)
        prev_period[name] = prev_period[name].fillna(0)
        
        this_period[f"{name}_change"] = this_period[name] - prev_period[name]
       
        # create a dummy for the type of portfolio that a person got extra 
        this_period[f"{name}_change_dummy"] = 0
        this_period.loc[this_period[f"{name}_change"]>0, f"{name}_change_dummy"]=1
        
    
    #------------ Select the variables for the final dataset ------------
    
    # We want the cases where there is an absolute increase in the number of portfolios,
    # And no decrease per type of portfolio. We additionally only look at cases
    # where a person gets only 1 of a particular type of portfolio
    
    select_portfoliogain =( (this_period['portfoliototaal']>prev_period['portfoliototaal']) \
                          & (this_period['business_change']>=0) \
                          & (this_period['business_change']<=1) \
                          & (this_period['retail_change']>=0) \
                          & (this_period['retail_change']<=1) \
                          & (this_period['joint_change']>=0) \
                          & (this_period['joint_change']<=1))
        
    data = this_period.loc[select_portfoliogain, ["percdiff", "portfolio_change",
                                                  "business_change", 
                                                  "retail_change","joint_change",
                                                  "business_change_dummy",
                                                  "retail_change_dummy",
                                                  "joint_change_dummy",]]

    data["saldo_prev"] = saldo_prev
    data["saldo_now"] = saldo_now
    
    # get some extra variables of interest from the preceding period
    
    data = data.reset_index(drop=True)
    previous_period = prev_period.loc[select_portfoliogain].reset_index(drop=True)
    
    if select_variables is not None:
         data[select_variables] = previous_period[select_variables]
          
    if dummy_variables is not None:
        #dummycolumns =  prev_period.loc[select_portfoliogain,
        #                                           dummy_variables]
        dummies,dummynames =  make_dummies(previous_period,
                                           dummy_variables) # use dummy variables as prefix
        data[dummynames] = dummies[dummynames]
        
        
    return data.reset_index(drop=True)
    



# =============================================================================
# Methods for aggregating and transforming
# =============================================================================


def getlogs(Y):
    minY = Y.min()
    print(f"Taking log of {Y.name}, minimum that gets added to log: {minY}")
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

    
    # Take the MAX of the logins, transactions and activity status 
    for var in ['logins_totaal','aantaltransacties_totaal','activitystatus']:
        df[f'{var}'] = df[[f"{var}_business",f"{var}_retail", 
                                f"{var}_joint"]].max(axis=1)

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
    for name in (["business","retail","joint","accountoverlay"]):
        df[f"{name}_max"] = df[f"{name}"]

        df.loc[(df[f"{name}"]>3),f"{name}_max"] = "3"   # Should I make it a string or make it a number?
        # This only relevant for business and joint since retail does not have >1
        
    # TODO also put a max on the total number of products???? or take the log??
    
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
    df["age_bins"] = pd.cut(df["age"], bins, labels=False)
    
    
    # also make bins out of the business age! mostly range between 0 and 30
    bbins = pd.IntervalIndex.from_tuples([ (-1, 2), (2, 5), 
                                         (5, 10), (20, 30),
                                         (30, np.inf)])
    df["businessAgeInYears_bins"] = pd.cut(df["businessAgeInYears"], bbins, labels=False)
    
    return df
    
    
    
def make_dummies(df, dummieslist, drop_first = False):
    """Returns a dataframe of dummies and the column names"""
    
    dummiesdf = df[dummieslist].astype('category')
    if drop_first:
        dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist,
                                   drop_first = True)
    else:
        dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist)
        
    return dummiesdf, list(dummiesdf.columns.values)
