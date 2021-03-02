"""
Contains extra functions to transform data

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from datetime import date
import utils
import gc
import declarationsFile


# =============================================================================
# Methods for making saldo prediction dataset
# =============================================================================
class AdditionalDataProcess(object):
    def __init__(self,indir, interdir, outdir, automatic_folder_change = False):
        self.indir = indir
        self.interdir = interdir
        self.outdir = outdir
        self.seed = 978391
        self.automatic_folder_change = False

        #Variable to change data folder based on selected first and last date
        ##Declare variables to provide checks for being empty
        self.input_cross = pd.DataFrame()
        self.input_cross = pd.DataFrame()
        self.panel_df = pd.DataFrame()
        self.cross_df = pd.DataFrame()
        self.cross_compared_df = pd.DataFrame()


    def create_saldo_data(self,dflist, interdir, filename ="saldodiff",
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
                data = self.get_difference_data(dfnew,dfold,
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

        diffdata["log_saldoprev"] = self.getlogs(diffdata["saldo_prev"])
        print(f"Done creating saldo prediction dataset at {utils.get_time()}" \
              f", saving to {filename}")
        utils.save_df_to_csv(diffdata, interdir, filename, add_time = False )
        return diffdata



    def get_difference_data(self,this_period,prev_period, log =True,
                            select_variables=None, dummy_variables =None, select_no_decrease = True):
        """Get the datapoints for which there is a difference of 1 or more
        portfolios"""
        #TODO also add account overlay somehow so that we can also find the impact
        # of that (if there is any??)

        # Get the total number of portfolios in each period (excluding overlay)

        #TODO wellicht dat dit het oplost?
        #Andere Oplossing is om de personid als index te nemen
        this_period.sort_values('personid', inplace = True)
        prev_period.sort_values('personid', inplace = True)
        this_period.reset_index(drop = True, inplace =  True)
        prev_period.reset_index(drop = True, inplace =  True)


        #TODO wat zijn de benodigdheden om dit te runnen
        #TODO schrijf dit om zodat dit verscihl per personid wordt gepakt
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
                                                      "retail_change","joint_change",
                                                      "business_change_dummy",
                                                      "retail_change_dummy",
                                                      "joint_change_dummy",]]

        data["saldo_prev"] = saldo_prev
        data["saldo_now"] = saldo_now

        # get some extra variables of interest from the preceding period

        data = data.reset_index(drop=True)

        previous_period = prev_period.loc[select_portfoliogain].reset_index(drop=True)

        # Add dummies for if we had business, retail or joint already in the last period
        for name in ["business","retail","joint"]:
            data[f"prev_{name}_dummy"] = previous_period[f"{name}_dummy"]

        # Add other dummies from the previous period which we gave as input
        if select_variables is not None:
             data[select_variables] = previous_period[select_variables]

        if dummy_variables is not None:
            #dummycolumns =  prev_period.loc[select_portfoliogain,
            #                                           dummy_variables]
            dummies,dummynames =  self.make_dummies(previous_period,
                                               dummy_variables) # use dummy variables as prefix
            data[dummynames] = dummies[dummynames]

        return data.reset_index(drop=True)



    # =============================================================================
    # Methods for aggregating and transforming
    # =============================================================================


    def getlogs(self,Y):
        minY = Y.min()
        print(f"Taking log of {Y.name}, minimum that gets added to log: {minY}")
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
        # also make a dummy
        for name in (["business","retail","joint","accountoverlay"]):
            df[f"{name}_max"] = df[f"{name}"]
            df.loc[(df[f"{name}"]>3),f"{name}_max"] = 3   # Should I make it a string or make it a number?
            # This only relevant for business and joint since retail does not have >1

            df[f"{name}_dummy"] = df[f"{name}"]
            df.loc[(df[f"{name}"]>0),f"{name}_dummy"] = 1

        # TODO also put a max on the total number of products???? or take the log??

        # handle it if there are still multiple categories in the gender
        df.loc[(df["geslacht"]=="Mannen"), "geslacht"]="Man"
        df.loc[(df["geslacht"]=="Vrouwen"), "geslacht"]="Vrouw"

        # make age into years
        today = date.today()
        df["age"] = today.year - df["birthyear"]
        # Create bins out of the ages
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



    def make_dummies(self, df, dummieslist, drop_first = False):
        """Returns a dataframe of dummies and the column names"""

        dummiesdf = df[dummieslist].astype('category')
        if drop_first:
            dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist,
                                       drop_first = True)
        else:
            dummiesdf = pd.get_dummies(dummiesdf, prefix = dummieslist)


        return dummiesdf, list(dummiesdf.columns.values)


    """
    Methods for data creation for general cross-section data and machine learning methods.
    To correctly run, just provide which transformation is warranted. It does not yet call upon the dataprocessor to 
    create the base data needed to use this algorithm however. 
    """

    def transform_to_different_sets( self, transform_command = "all", first_date = "", last_date = "" ):
        if first_date == "":
            use_standard_period = True
        else:
            use_standard_period = False

        if transform_command in ['cross_df', 'all']:
            if use_standard_period:
                first_date, last_date = "2019Q1","2019Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_cross_data()

        if transform_command in ['cross_long_df', 'all']:
            if use_standard_period:
                first_date, last_date = "2018Q1","2020Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_cross_data(last_date)
            self.add_longterm_change(benchmark_period = first_date)

        if transform_command in ['panel_df', 'all']:
            if use_standard_period:
                first_date, last_date = "2018Q1","2020Q4"
            self.prepare_before_transform(first_date, last_date)
            self.transform_for_panel()


    def set_dates( self, first_date, last_date, override_folder_change = False ):
        "Method for safely changing dates in this class"
        data_freq = utils.infer_date_frequency(first_date)
        assert data_freq != None \
               and (utils.infer_date_frequency(last_date) != None or \
                    last_date == ""), \
            "No Valid date set"
        self.first_date = first_date
        self.current_freq = data_freq

        if last_date == "":
            self.last_date = first_date
            self.multiple_periods_imported = False
            print(f"Period set to {self.first_date} with frequency {self.current_freq}")
        else:
            self.last_date = last_date
            self.multiple_periods_imported = True
            print(f"Period set from {self.first_date} to {self.last_date} with frequency {self.current_freq}")

        if self.automatic_folder_change and not override_folder_change:
            self.folder_operations("replace_time",self.first_date,self.last_date)

    ###------LAST TRANSFORMATION TO DATA BEFORE MODELLING ----------###
    def prepare_before_transform( self, first_date = "", last_date = "" ):
        print(f"Starting preparing data at {utils.get_time()}")
        if first_date == "":
            self.set_dates("2018Q1", "2020Q4")
        else:
            self.set_dates(first_date, last_date)

        #Import the different files related to the data to be prepared for transformation
        self.import_data('input_cross', first_date = self.first_date, last_date = self.last_date)

        #Transform the different dataframes that have been imported
        for i, df in enumerate(self.input_cross_list):
            new_df = self.aggregate_portfolio_types(df)
            # new_df = self.transform_variables(new_df)
            # if i == 0:
            #     prev_df = new_df
            # else:
            #     diffdata = self.get_difference_data(new_df, prev_df, log = False, select_no_decrease =  False)
            #     if i == 1:
            #         filter_diffdata = ['personid'] + list(set(diffdata.columns) - set(new_df.columns))
            #
            #     prev_df = new_df
            #     new_df = pd.merge(new_df, diffdata[filter_diffdata], how = "left", on = 'personid')
            #     new_df.reset_index()
            self.input_cross_list[i] = new_df



        self.input_cross = pd.concat(self.input_cross_list, ignore_index = True)

        # Make lowercase names
        self.input_cross.rename(str.lower, axis = 'columns', inplace = True)

        #Fill NA values with zeros to correctly set observations with not one of these options
        fillna_columns = ['business', 'joint', 'retail', 'accountoverlay', 'accountoverlay_dummy', 'accountoverlay_max']
        fillna_columns = utils.doListIntersect(fillna_columns,self.input_cross.columns)
        self.input_cross[fillna_columns] = self.input_cross[fillna_columns].fillna(value = 0)

        rename_dict = {
            # 'retail_dummy'  : 'has_ret_prtf',
            # 'joint_dummy'   : 'has_jnt_prtf',
            # 'business_dummy': 'has_bus_prtf',
            # 'accountoverlay_dummy' : 'has_accountoverlay',
            'business': 'business_prtf_counts',
            'joint' : 'joint_prtf_counts',
            'retail': 'retail_prtf_counts',
            'accountoverlay': 'aantalproducten_accountoverlay',
            # 'accountoverlay_dummy': 'has_accountoverlay',
            # 'portfoliototaal': 'portfolio_total_counts'
        }
        rename_dict = utils.doDictIntersect(self.input_cross.columns, rename_dict)
        self.input_cross.rename(rename_dict, axis = 1, inplace = True)

        today = date.today()
        self.input_cross = self.input_cross.assign(
            period_q2 = lambda x: np.where(x.period_obs.dt.quarter == 2, 1, 0),
            period_q3 = lambda x: np.where(x.period_obs.dt.quarter == 3, 1, 0),
            period_q4 = lambda x: np.where(x.period_obs.dt.quarter == 4, 1, 0),
            period_year = lambda x: x.period_obs.dt.year,
            has_ret_prtf = lambda x: np.where(x.retail_prtf_counts > 0,1,0),
            has_jnt_prtf = lambda x: np.where(x.joint_prtf_counts > 0,1,0),
            has_bus_prtf = lambda x: np.where(x.business_prtf_counts > 0,1,0),
            has_accountoverlay =  lambda x: np.where(x.aantalproducten_accountoverlay > 0,1,0),
            portfolio_total_counts = lambda x: x.business_prtf_counts + x.joint_prtf_counts + x.retail_prtf_counts,
            current_age = lambda x:today.year - x.birthyear
        )

        self.input_cross.loc[(self.input_cross["geslacht"]=="Mannen"), "geslacht"]="Man"
        self.input_cross.loc[(self.input_cross["geslacht"]=="Vrouwen"), "geslacht"]="Vrouw"

        # Drop unnecessary columns
        list_to_drop = ['valid_to_dateeow', 'valid_from_dateeow', 'valid_from_min', 'valid_to_max', 'saldototaal_agg']
        list_to_drop = utils.doListIntersect(list_to_drop, self.input_cross.columns)
        self.input_cross.drop(list_to_drop, axis = 1, inplace = True)

        self.input_cross.sort_index(inplace = True, axis = 1)

        print(f"Finished preparing data at {utils.get_time()}")

    def transform_for_cross_data( self, date_for_slice = "2019Q4" ):
        """
        Method to perform some transformations to prepare a cross-section.
        Selected variables are chosen to be take a mean over the year.
        Method to impute missing values to more correctly balance
        """

        search_list_counts = ['aantalatmtrans','aantalbetaaltrans','aantalfueltrans','aantallogins','aantalposttrans',
                              'aantaltegenrek','aantaltrans', 'logins_']
        exclusion_list_counts = ['bins']
        search_list_balance = ['saldototaal']
        exclusion_list_balance = []

        columns_to_use_list = utils.do_find_and_select_from_list(self.input_cross.columns, search_list_counts,
                                                             exclusion_list_counts)
        columns_to_use_list = columns_to_use_list + \
                              utils.do_find_and_select_from_list(self.input_cross.columns,search_list_balance,exclusion_list_balance)
        indicators_list = ['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']
        columns_to_use_list = ['personid', 'period_obs'] + indicators_list + columns_to_use_list


        cross_df = self.input_cross[columns_to_use_list].copy()
        time_conv_dict = {"Q": 4, "M": 12, "W": 52, "Y": 1}
        period_list = pd.period_range(end = date_for_slice, periods = time_conv_dict[self.current_freq], freq = self.current_freq)
        cross_df = cross_df[cross_df.period_obs.isin(period_list)]
        cross_df.sort_values(['period_obs', 'personid'], inplace = True)

        incomplete_obs = cross_df.groupby('personid', as_index = False)[indicators_list].sum()
        incomplete_obs_person_bus = incomplete_obs.query("0 < has_bus_prtf < 4").personid
        incomplete_obs_person_jnt = incomplete_obs.query("0 < has_jnt_prtf < 4").personid
        incomplete_obs_person_ret = incomplete_obs.query("0 < has_ret_prtf < 4").personid

        incomplete_person_list = set(incomplete_obs_person_bus) | set(incomplete_obs_person_jnt) | set(incomplete_obs_person_ret)

        incomplete_index = cross_df.personid.isin(incomplete_person_list)
        incomplete_df = cross_df[incomplete_index]
        cross_df = cross_df[~incomplete_index]  # Select complete observations for now

        mean_values = cross_df.groupby('personid').mean()
        cross_df.set_index(["period_obs", "personid"], inplace = True)

        indexed_df = pd.DataFrame(columns = period_list, index = cross_df.columns)
        for period in period_list:
            indexed_result = cross_df.loc[period] / mean_values
            indexed_df[period] = indexed_result.mean()
        indexed_df = indexed_df.transpose()
        cross_df.reset_index(inplace = True)

        ##LARGE OPERATION TO MATCH ALL COLUMNS WHICH ARE DEPENDENT ON THE SECTOR
        # business_columns = column_loop(cross_df, "business")
        business_columns = utils.do_find_and_select_from_list(cross_df.columns,['business'])
        retail_columns = utils.do_find_and_select_from_list(cross_df.columns,['retail'])
        joint_columns = utils.do_find_and_select_from_list(cross_df.columns,['joint'])
        standard_cols = ['personid', 'period_obs']
        incomplete_final = incomplete_df[standard_cols]

        """-INTERPOLATE THE MISSING VALUES BASED ON THE GENERAL INDEX DERIVED ABOVE AND THE FIRST AVAILABLE QUANTITY 
        OF THE DIFFERENT PERSONIDS WITH MISSING OBSERVATIONS"""
        outer_merge_list = []
        for persons, cols, indic in [incomplete_obs_person_bus, business_columns, 'has_bus_prtf'], \
                                    [incomplete_obs_person_jnt, joint_columns, 'has_jnt_prtf'], \
                                    [incomplete_obs_person_ret, retail_columns, 'has_ret_prtf']:

            incomplete_df_slice = incomplete_df[incomplete_df.personid.isin(persons)]
            incomplete_df_slice_persons = incomplete_df_slice.groupby(['personid'], as_index = False).apply(lambda x: x.loc[
                x[indic] == 1, 'period_obs'].min())
            incomplete_df_slice_persons = incomplete_df_slice_persons.rename({None: "benchmark_period"}, axis = 1)
            incomplete_df_slice = pd.merge(incomplete_df, incomplete_df_slice_persons, on = 'personid')

            cols_without_parameters = standard_cols + ['benchmark_period', indic]
            cols_complete = cols_without_parameters + cols

            incomplete_df_slice = incomplete_df_slice[cols_complete]
            indexed_df_slice = indexed_df[cols]  # Take a slice of the variable that has previously been created to index

            inner_df_list = []
            outer_df_list = []

            for period_outer in period_list:
                templist = (incomplete_df_slice[indic] == 0) & (incomplete_df_slice['period_obs'] == period_outer)

                # If a part has been defined in the period it will not be analyzed further and added to the end result.
                correct_list = (incomplete_df_slice[indic] == 1) & (incomplete_df_slice['period_obs'] == period_outer)
                outer_df = incomplete_df_slice.loc[correct_list, cols_complete]
                outer_df = outer_df.drop('benchmark_period', axis = 1)
                outer_df_list.append(outer_df)

                for period_inner in period_list:
                    templist2 = templist & (incomplete_df_slice['benchmark_period'] == period_inner)
                    if templist2.sum() > 0:
                        templist3 = (incomplete_df_slice.personid.isin(incomplete_df_slice[templist2].personid)) & \
                                    (incomplete_df_slice.period_obs == period_inner)
                        inner_df = incomplete_df_slice.loc[templist3, cols] \
                                   * indexed_df_slice.loc[period_inner]
                        inner_df = pd.concat([incomplete_df_slice.loc[templist2, cols_without_parameters].reset_index(
                            drop = True), inner_df.reset_index(drop = True)], axis = 1, ignore_index = True)
                        inner_df.columns = cols_complete
                        inner_df = inner_df.drop("benchmark_period", axis = 1)
                        inner_df_list.append(inner_df)

            inner_and_outer = outer_df_list + inner_df_list
            outer_merge_list.append(pd.concat(inner_and_outer, ignore_index = True))

        for item in outer_merge_list:
            incomplete_final = pd.merge(incomplete_final, item, how = "left", on = standard_cols)
        incomplete_final.sort_index(axis = 1, inplace = True)

        incomplete_final[['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']] = incomplete_final[
            ['has_bus_prtf', 'has_jnt_prtf', 'has_ret_prtf']].fillna(value = 0)

        remaining_vars = list(set(incomplete_df.columns) - set(incomplete_final.columns))
        remaining_vars = standard_cols + remaining_vars


        print(f"The final dataset has dim :{incomplete_final.shape} and the one before transforming has {incomplete_df.shape}")
        incomplete_final = pd.merge(incomplete_final, incomplete_df[remaining_vars], on = standard_cols)
        print(f"With final result : {incomplete_final.shape}. With indicator variables still in, which are discarded before the "
              f"final merge with the larger set.")

        # Concat to the larger set without NaN
        cross_df = pd.concat([cross_df, incomplete_final], ignore_index = True)

        # Clean variables
        del incomplete_df, incomplete_df_slice, incomplete_df_slice_persons, incomplete_final, incomplete_index, \
            incomplete_obs, incomplete_obs_person_ret, incomplete_obs_person_jnt, incomplete_obs_person_bus, incomplete_person_list, \
            indexed_df_slice, inner_df, inner_df_list, inner_and_outer, outer_merge_list, outer_df, outer_df_list, period_inner, \
            period_outer, templist, templist2, templist3
        gc.collect()

        cross_df.drop(indicators_list, axis = 1, inplace = True)
        cross_df = cross_df.groupby("personid").mean().reset_index()

        remaining_vars = list(set(self.input_cross) - set(cross_df))
        remaining_vars = ['personid'] + remaining_vars

        #Takes last period and merges this to the variables who have just gotten a mean value
        partial_input_cross = self.input_cross.loc[self.input_cross.period_obs == period_list[-1], remaining_vars]
        self.cross_df = pd.merge(cross_df, partial_input_cross, on = 'personid')
        print(f"Finished larger merge with final dimensions :{cross_df.shape}")

        print(f"Finished transforming data for cross section {utils.get_time()}")

    def add_longterm_change(self, benchmark_period):
        """"
        Uses cross section, however does add variables which compare a certain historical point with
        the latest date.
        """
        product_counts_list = [
            'aantalproducten_totaal',
            'aantalproducten_totaal_business',
            'aantalproducten_totaal_joint',
            'aantalproducten_totaal_retail',
            'aantalproducten_accountoverlay',
            'portfolio_total_counts',
            'business_prtf_counts',
            'joint_prtf_counts',
             'retail_prtf_counts',
        ]

        benchmark_period = pd.to_datetime(benchmark_period).to_period(self.current_freq)

        benchmark_slice = self.input_cross.query(f"period_obs == @benchmark_period")
        benchmark_slice = benchmark_slice[ ( ['personid'] + product_counts_list) ]

        self.cross_long_df = self.cross_df[(['personid'] + product_counts_list)].copy()
        self.cross_long_df.set_index('personid'), benchmark_slice.set_index('personid')

        self.cross_long_df.set_index('personid', inplace = True)
        benchmark_slice.set_index('personid', inplace = True)
        self.cross_long_df.sort_index(inplace = True)
        benchmark_slice.sort_index(inplace = True)

        indicator_df1 = self.cross_long_df - benchmark_slice

        indicator_df2 = indicator_df1.where(indicator_df1 > 0, 0)
        indicator_df2 = indicator_df2.where(indicator_df1 == 0, 1)

        indicator_df1 = indicator_df1.add_prefix('delta_')
        indicator_df2 = indicator_df2.add_prefix('increased_')
        benchmark_slice = benchmark_slice.add_prefix('benchmark_')

        df_to_merge = pd.concat([indicator_df1,indicator_df2,benchmark_slice], axis = 1)
        self.cross_long_df.sort_index(axis = 1,inplace = True)
        self.cross_long_df = pd.merge(self.cross_df, df_to_merge, left_on = 'personid', right_index =  True)


        pass

    def transform_for_panel( self ):
        """
        Calculates how much selected variables have changed and creates an indicator if this change is positive.
        First loops over the different time periods and creates change variables compared to the previous period.
        After that creates an indicator value where this value is positive
        Last step is merging it back to the larger self.prepared_df dataset
        """
        templist = [
            'personid',
            'period_obs',
            'aantalproducten_totaal',
            'aantalproducten_totaal_business',
            'aantalproducten_totaal_joint',
            'aantalproducten_totaal_retail',
            'aantalproducten_accountoverlay',
            'business_prtf_counts',
            'joint_prtf_counts',
            'retail_prtf_counts',
            'portfolio_total_counts'
        ]

        delta_df = self.input_cross[templist].copy()
        delta_df = delta_df.set_index(['period_obs', 'personid'])
        delta_df.sort_index(inplace = True)

        frame_list = []
        period_index = pd.period_range(start = self.first_date, end = self.last_date, freq = self.current_freq)
        for current_date_in_loop in period_index[1:]:
            new_delta_frame = delta_df.loc[current_date_in_loop,:] - delta_df.loc[current_date_in_loop - 1,:]
            new_delta_frame = new_delta_frame.reset_index()
            new_delta_frame['period_obs'] = current_date_in_loop
            frame_list.append(new_delta_frame)
        new_delta_frame = pd.concat(frame_list, ignore_index = True)

        templist = list(set(new_delta_frame.columns) - set(['period_obs', 'personid']))
        new_delta_frame[templist] = np.where(new_delta_frame[templist] > 1, 1, 0)

        self.panel_df = pd.merge(self.input_cross, new_delta_frame, on = ['period_obs', 'personid'],
                                 suffixes = ["", "_delta"])

        print("number of positively changed variables is :\n", self.panel_df.iloc[:, -len(templist):].sum(), f"\nFrom a total of" \
                                                                                                         f" {self.panel_df.shape[0]} observations")
        #TODO Onderzoek 035bafd486c3223b24e1e76008847b64fb6c3cdc,
        print(f"Finished aggregating data for change in products at {utils.get_time()}")

    def import_data( self, import_string: str, first_date: str, last_date = "", addition_to_file_name = "" ):

        if last_date != "":
            exec(f"self.{import_string}_list = []")
            self.current_freq = utils.infer_date_frequency(first_date)
            for current_date_in_loop in pd.period_range(start = first_date, end = last_date, freq = self.current_freq):
                self.import_data(import_string, first_date = current_date_in_loop, addition_to_file_name = addition_to_file_name)
                exec(f"{import_string}_{current_date_in_loop} = self.{import_string}.copy()")
                exec(f"{import_string}_{current_date_in_loop}['period_obs'] = current_date_in_loop")
                exec(f"self.{import_string}_list.append({import_string}_{current_date_in_loop})")

        else:
            if import_string == 'input_cross':
                self.input_cross = pd.read_csv(f"{self.interdir}/final_df_{first_date}{addition_to_file_name}.csv")

    def folder_operations(self, folder_command, first_date = None, last_date = None, keywords_list = None, **extra_args):
        if (first_date == None) and (folder_command in ['create_sub_and_import','replace_time']):
            first_date = self.first_date
            last_date = self.last_date
        else:
            if last_date == None:
                last_date = first_date

        ##Importing different sets of data
        if keywords_list == None:
            keywords_list = ['final_df', 'valid_id', 'portfoliolink', 'base_experian', 'base_linkinfo']

        if folder_command == "create_sub_and_import":
            utils.create_subfolder_and_import_files(first_date =  first_date,last_date = last_date,subfolder = self.interdir,
                                                    find_list = keywords_list ,**extra_args)
        elif folder_command == "replace_time":
            utils.replace_time_period_folder(first_date =  first_date, last_date = last_date, subfolder = self.interdir,
                                             remove_list = keywords_list,**extra_args)
        elif folder_command == 'clean_folder':
            utils.replace_time_period_folder(subfolder = self.interdir,remove_list = keywords_list,**extra_args)
        else:
            print("Wrong Value: Choose either |'final_df', 'create_sub_and_import','clean_folder' |")

    def debug_in_class( self ):
        "Method to be able to perform operations as if debugging in class method"
        print("hey")
