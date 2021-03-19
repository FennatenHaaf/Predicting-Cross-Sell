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
class AdditionalDataProcess(object):
    def __init__(self,indir, interdir, outdir, automatic_folder_change = False):
        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir
        self.seed = 978391

    # 2 methodes: create full difference en in de main een get differences tussen
    # 2 specifieke periodes??
    def create_crosssell_data(self,dflist, interdir, filename ="crossselloverall"):
        """Get a dataset containing all portfolio increases, decreases and
        non-changes over all periods in dflist"""
        i=0
        for df in dflist:
            if i==0:      
                dfold = df
            else:
                dfnew = df
                data = self.get_difference_data(dfnew,dfold,
                                           select_variables = None,
                                           dummy_variables = None,
                                           select_no_decrease = False,
                                           globalmin = None)
                if i==1:
                    diffdata = data
                else:
                    diffdata = pd.concat([diffdata,data], ignore_index= True) 
                    diffdata = diffdata.fillna(0)

                dfold = dfnew
            i+=1
            
        # Select only the information that we have about portfolio changes
        diffdata = diffdata[["personid", "portfolio_change",
                            "business_change", "retail_change",
                            "joint_change","accountoverlay_change",
                            "business_change_dummy", "retail_change_dummy",
                            "joint_change_dummy","accountoverlay_change_dummy"]]
        
        print(f"Done creating cross-sell differences dataset at {utils.get_time()}" \
              f", saving to {filename}")
        utils.save_df_to_csv(diffdata, interdir, filename, add_time = False )
        return diffdata
        

    def create_saldo_data(self,dflist, interdir, filename ="saldodiff",
                          select_variables = None, dummy_variables = None,
                          globalmin = None):
        """Make saldo data out of a df list"""
        
        # We assume our input is already transformed & aggregated!
        i=0
        for df in dflist:
            if i==0:      
                dfold = df
            else:
                dfnew = df
                data = self.get_difference_data(dfnew,dfold,
                                           select_variables = select_variables,
                                           dummy_variables = dummy_variables,
                                           select_no_decrease = True,
                                           globalmin = globalmin)
                if i==1:
                    diffdata = data
                else:
                    diffdata = pd.concat([diffdata,data], ignore_index= True) #
                    diffdata = diffdata.fillna(0)

                dfold = dfnew
            i+=1

        diffdata["log_saldoprev"] = self.getlogs(diffdata["saldo_prev"])
        print(f"Done creating saldo prediction dataset at {utils.get_time()}" \
              f", saving to {filename}")
        utils.save_df_to_csv(diffdata, interdir, filename, add_time = False )
        return diffdata



    def get_difference_data(self,this_period,prev_period, log =True,
                            select_variables=None, dummy_variables =None,
                            select_no_decrease = True, globalmin = None,
                            verbose = True):
        """Get the datapoints for which there is a difference of 1 or more
        portfolios"""

        # Sorteer voor de zekerheid, andere Oplossing is om personid als index te nemen
        this_period.sort_values('personid', inplace = True)
        prev_period.sort_values('personid', inplace = True)
        this_period.reset_index(drop = True, inplace =  True)
        prev_period.reset_index(drop = True, inplace =  True)

        # Get the total number of portfolios in each period (excluding overlay)
        #TODO schrijf dit om zodat dit verscihl per personid wordt gepakt?
        this_period['portfoliototaal']= this_period[["business","retail",
                                            "joint"]].sum(axis=1)
        prev_period['portfoliototaal']= prev_period[["business","retail",
                                            "joint"]].sum(axis=1)

        this_period['portfolio_change'] = this_period['portfoliototaal'] - \
                                          prev_period['portfoliototaal']

        saldo_prev = prev_period['saldototaal']
        saldo_now = this_period['saldototaal']
        
        if not( isinstance(globalmin, type(None)) ):
            minoverall = globalmin
        else:
            minoverall = min(saldo_prev.min(),saldo_now.min())

        if log: 
            # Need to subtract the same number of each to take the log
            # and then get the log difference
            #print(f"Taking log difference, min of both is {minoverall}")
            logprev = np.log(saldo_prev+1-minoverall)
            lognow = np.log(saldo_now+1-minoverall)

            this_period['percdiff'] =lognow-logprev
        else:
            # Take the percentage - we add the minimum of all periods to
            # deal with negative or zero values
            #print("Taking percentage difference, min of both is {minoverall}")
            saldo_prev2 = this_period['saldototaal']+1+minoverall
            saldo_now2 = prev_period['saldototaal']+1+minoverall
            this_period['percdiff'] =((saldo_now2-saldo_prev2) / saldo_prev2)*100

        # Get portfolio variables
        for name in (["business","retail","joint", "accountoverlay"]):
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
        # WE ALSO INCLUDE PEOPLE WHO DON'T HAVE A CHANGE IN THE NUMBER OF PORTFOLIOS
        if select_no_decrease:
            select_portfoliogain =( (this_period['portfoliototaal']>=prev_period['portfoliototaal']) \
                                  & (this_period['business_change']>=0) \
                                  & (this_period['business_change']<=1) \
                                  & (this_period['retail_change']>=0) \
                                  & (this_period['retail_change']<=1) \
                                  & (this_period['joint_change']>=0) \
                                  & (this_period['joint_change']<=1))
        else:
            select_portfoliogain = np.full((this_period.shape[0],1), True)

        data = this_period.loc[select_portfoliogain, ["personid", "percdiff", 
                                                      "portfolio_change",
                                                      "business_change",
                                                      "retail_change",
                                                      "joint_change",
                                                      "accountoverlay_change",
                                                      "business_change_dummy",
                                                      "retail_change_dummy",
                                                      "joint_change_dummy",
                                                      "accountoverlay_change_dummy"]]
        
        data["saldo_prev"] = saldo_prev
        data["saldo_now"] = saldo_now

        # get some extra variables of interest from the preceding period
        data = data.reset_index(drop=True)
        previous_period = prev_period.loc[select_portfoliogain].reset_index(drop=True)

        # Add dummies for if we had business, retail or joint already in the last period
        for name in ["business","retail","joint"]:
            data[f"prev_{name}_dummy"] = previous_period[f"{name}_dummy"]

        # Add other dummies from the previous period which we gave as input
        if not( isinstance(select_variables, type(None)) ):
             data[select_variables] = previous_period[select_variables]

        if not( isinstance(dummy_variables, type(None)) ): 
            dummies,dummynames =  self.make_dummies(previous_period,
                                               dummy_variables) # use dummy variables as prefix
            data[dummynames] = dummies[dummynames]

        return data.reset_index(drop=True)



# =============================================================================
# Methods for aggregating and transforming
# =============================================================================


    def getlogs(self,Y):
        minY = Y.min()
        #print(f"Taking log of {Y.name}, minimum that gets added to log: {minY}")
        # add the smallest amount of Y
        logY = np.log(Y+1-minY)
        return logY



    def aggregate_portfolio_types(self,df):
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


    def transform_variables(self,df, separate_types = False):
        """Transform variables so that they are all around the same scale"""

        # Take the LOGARITHM of the account details, logins and transactions!
        for var in (["saldototaal","logins_totaal","aantaltransacties_totaal"]):
            if (separate_types):
                for name in (["business","retail","joint"]):
                    df[f"log_{var}_{name}"] = self.getlogs(df[f"{var}_{name}"])

            df[f"log_{var}"] =  self.getlogs(df[f"{var}"])
            # also make a version of this variable where they are divided into bins
            df[f"log_{var}_bins"] = pd.cut(df[f"log_{var}"], bins=3, labels=False)

        #also take bins for aantalproducten totaal
        df[f"aantalproducten_totaal_bins"] = pd.cut(df[f'aantalproducten_totaal'], 
                                                    bins=3, labels=False)

        # Put a MAX on the business, portfolio, joint variables (3 per type max)
        for name in (["business","retail","joint","accountoverlay"]):
            df[f"{name}"] = df[f"{name}"].fillna(0)
            df[f"{name}_max"] = df[f"{name}"]
            
            # also make a dummy for if there is more than 0
            df[f"{name}_dummy"] = df[f"{name}"]
            df.loc[(df[f"{name}"]>0),f"{name}_dummy"] = 1
            
            if (name=="business"): 
                df.loc[(df[f"{name}"]>3),f"{name}_max"] = 3   
                # This only relevant for business since retail does not have >1
            elif (name=="accountoverlay"):
                df.loc[(df[f"{name}"]>2),f"{name}_max"] = 2   
            else: # for joint and retail we consider 1 the maximum
                df.loc[(df[f"{name}"]>1),f"{name}_max"] = 1   

        # handle it if there are still multiple categories in the gender
        df.loc[(df["geslacht"]=="Mannen"), "geslacht"]="Man"
        df.loc[(df["geslacht"]=="Vrouwen"), "geslacht"]="Vrouw"

        # make age into years
        today = date.today()
        df["age"] = today.year - df["birthyear"]
        # Create bins out of the ages
        bins = pd.IntervalIndex.from_tuples([ (0, 18), (18, 30),
                                             (30, 40), (40, 50),
                                             (50, 65), (65, 200)])
        df["age_bins"] = pd.cut(df["age"], bins, labels=False)

        # also make bins out of the business age! mostly range between 0 and 30
        bbins = pd.IntervalIndex.from_tuples([ (-1, 3), (3, 6),
                                             (6, 12), (12, 25),
                                             (25, np.inf)])
        df["businessAgeInYears_bins"] = pd.cut(df["businessAgeInYears"], bbins, labels=False)
        
        # finally, make bins out of saldo
        bins = pd.IntervalIndex.from_tuples([ (-np.inf, 0), (0, 100),
                                             (100, 1000), (1000, 5000),
                                             (5000, 15000), (15000, 50000),
                                             (50000, np.inf)])
        df["saldototaal_bins"] = pd.cut(df["saldototaal"], bins, labels=False)
        

        # transform to 'other' categories
        #self.make_other_category(df,"businessType",limit=5)
        df["businessType"] = df["businessType"].replace("Maatschap", "Maatschap/Stichting")
        df["businessType"] = df["businessType"].replace("Stichting", "Maatschap/Stichting")
        df["businessType"] = df["businessType"].replace("Besloten vennootschap", "Besloten Vennootschap")
      
        
        # Fix some categories for the businessType
        df["hh_size"] = df["hh_size"].replace(10.0, 1.0)
        df["hh_size"] = df["hh_size"].replace(11.0, 1.0)

        return df



    def make_dummies(self, df, dummieslist, drop_first = False):
        """Returns a dataframe of dummies and the column names"""

        dummiesdf = df[dummieslist].astype('category')
        if drop_first:
            dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist,
                                       drop_first = True)
        else:
            dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist)


        return dummiesdf, list(dummiesdf.columns.values)


    def make_other_category(self,df,cat_var,limit):
        """For a categoral variable, changes all values which appear
        fewer times than the limit to a category 'other' """
        
        df[f"{cat_var}_other"] = df[cat_var]
        g = df.groupby(cat_var)[cat_var].transform('size')
        df.loc[g < limit, f"{cat_var}_other"] = 'Anders'
        
        #TODO: als je dit per df apart doet zorgt het ervoor dat verschillende
        # categorieen per tijdsperiode in de 'anders' categorie komen


