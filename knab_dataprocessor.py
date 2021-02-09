# -*- coding: utf-8 -*-
"""
This code aims to create the data file that links portfolios to people

@author: Fenna ten Haaf
"""
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import utils
import dataInsight
import declarationsFile
import gc


class dataProcessor:

    def __init__(self,indir,interdir,outdir, 
                 save_intermediate=False,
                 print_info=False,
                 seed = 978391):
        """This method processes data provided by Knab"""
        
        #-------------------------INITIALISATION-----------------------
        self.indir= indir # location of input files
        self.interdir = interdir #location of intermediate files
        self.outdir = outdir # location of linked output files
        self.save_intermediate = save_intermediate # location of linked output files
        self.print_info = print_info #Determines how verbose it is
        self.seed = seed
        self.endDate = datetime(2020, 12, 31) #Initalizes a last date for NaN values

        #Declare data variables to check if data has been printed
        self.df_corporate_details = pd.DataFrame()
        self.df_pat = pd.DataFrame()
        self.df_expTS = pd.DataFrame()
        self.df_pat_sample = pd.DataFrame()
        
        
        
               
    def link_data(self, outname = "base_linkinfo"):
        """This function creates a dataset containing person IDs linked
        to their portfolio ids and the corresponding portfolio information.
        It also links the person IDs to business IDs and corresponding 
        business information based on the corporate portfolio IDs"""
        
        print(f"****Processing data, at {utils.get_time()}****")
    
        #---------------------READING AND MERGING RAW DATA---------------------
        self.df_experian = pd.read_csv(f"{self.indir}/experian.csv")                     
        
        if self.df_corporate_details.empty:
            self.processCorporateData() # create processed corporate dataset

        # Do a left join to add portfolio information  
        self.df_link = pd.read_csv(f"{self.indir}/linkpersonportfolio.csv").merge(
            pd.read_csv(f"{self.indir}/portfolio_info.csv"), 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 

        #Split the corporate persons off and rename
        df_corporatelink = self.df_link[self.df_link["iscorporatepersonyn"]==1]
        df_corporatelink = df_corporatelink.loc[:,["personid","portfolioid"]]
        df_corporatelink = df_corporatelink.rename(columns={"personid": "corporateid",})
        
        # Merge to find which human personids are linked to which corporate ids
        self.df_link = self.df_link.merge(df_corporatelink, 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 
        # TODO check if human person ids could be getting linked to multiple
        # corporate ids?
        
        # Merge the file with business information
        self.df_corporate_details = self.df_corporate_details.rename(columns={"personid": "corporateid"})
        self.df_link = self.df_link.merge(self.df_corporate_details, 
                                  how="left", left_on=["corporateid"],
                                  right_on=["corporateid"],)
        # TODO check if businesses appear multiple times!!
    
        
        #------------------------ GET DATES INSIGHT -------------------------
        
        # pd.to_datetime does not work on a DF, only a series or list, so we use .astype()
        self.df_experian.loc[:,['valid_to_dateeow']] = self.df_experian.loc[:,['valid_to_dateeow']].astype('datetime64[ns]')
        self.df_experian.loc[:,['valid_from_dateeow']] = self.df_experian.loc[:,['valid_from_dateeow']].astype('datetime64[ns]')

        # See what the most recent date is in the data 
        # Find the latest and oldest dates where a customer was changed or added 
        all_dates = pd.concat([self.df_experian['valid_to_dateeow'],self.df_experian['valid_from_dateeow']])
        self.last_date = all_dates.max()
        self.first_date = all_dates.min()
        
        # Print the results
        time_string = self.last_date.strftime("%Y-%m-%d")
        print(f"most recent data in Experian is from {time_string},")
        time_string = self.first_date.strftime("%Y-%m-%d")
        print(f"Oldest data in Experian is from {time_string}")
        
        
        #------------------------ SAVE & RETURN  -------------------------
        if self.save_intermediate:
            utils.save_df_to_csv(self.df_link, self.interdir, 
                                  outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")
        #return self.df_link 
       
        
       
        
    def select_ids(self, quarterly = True, subsample = True,
                   sample_size = 500, start_date = "2018", end_date = None,
                   outname = "base_experian"):
        """Selects a certain set of person IDs from the linking dataset, where:
            - information for ALL portfolios is present in the linking data
            - There is information for them in the Experian data for
            each time period (if time_series is TRUE) .
          We also take a subsample, if this is so specified."""

        if (end_date==None):
            end_date= self.last_date # Make this the end date period
        
        #--------- make sure there is information in the portfolio info data ---------
        
        # we only want those ids for which the portfolio information is present, 
        # so dateinstroomweek should NOT be blank
        valid_ids_link = self.df_link["personid"][~(self.df_link["dateinstroomweek"].isnull())]
        #TODO dit is nog niet alles, alleen nog maar AT LEAST one?
    
    
        #--------- make sure there is information in the Experian data ---------
        # we want there to be experian data in every time period for the IDs 
    
        # Get a variable representing the last date until which an ID is valid
        maxending = self.df_experian[["valid_to_dateeow","personid"]].copy()
        maxending["valid_to_dateeow"] = maxending["valid_to_dateeow"].fillna(self.last_date).astype('datetime64[ns]')
        maxending = maxending.groupby("personid").max()
        maxending = maxending.rename(columns={"valid_to_dateeow": "valid_to_max",})
        # add to experian data        
        self.df_experian = self.df_experian.merge(maxending, how="left", left_on=["personid"],
                               right_on=["personid"])
        
        # Get a variable representing the first date from which an ID is valid
        minstart = self.df_experian[["valid_from_dateeow","personid"]]
        minstart = minstart.groupby("personid").min()
        minstart  = minstart.rename(columns={"valid_from_dateeow": "valid_from_min",})                                 
        # add to experian data
        self.df_experian = self.df_experian.merge(minstart, how="left", left_on=["personid"],
                               right_on=["personid"])

        #-> The first validfrom date has to be BEFORE a certain starting date
        #-> The last validto date has to be AFTER a certain ending date    
        valid_ids_experian = self.df_experian["personid"][(self.df_experian["valid_to_max"]>= end_date) \
                                            & (self.df_experian["valid_from_min"]<= start_date)]        
        #TODO: zou het kunnen dat IDs in de tussentijd worden uitgeschreven en 
        # weer ingeschreven??
        
        # Get the intersection of both series
        valid_ids= valid_ids_experian[valid_ids_experian.isin(valid_ids_link)]
        
        # Make sure there are also the household size variables etc. present, 
        # not just finergy
        valid_ids_info = self.df_experian["personid"][~(self.df_experian["age_hh"].isnull())]
        # Get the intersection
        valid_ids= valid_ids[valid_ids.isin(valid_ids_info)]
               
        #---------------- Take a subsample of the IDs -----------------------
        
        if subsample:
            print(f"****Taking a subsample of {sample_size} IDs, at {utils.get_time()}.****")

            # Take a random subsample, seed is to get consistent results
            valid_ids = valid_ids.sample(n = sample_size, 
                                         random_state = self.seed).reset_index(drop=True)
            print(f"Done at {utils.get_time()}.")
            #TODO nu nog ergens valid IDs invullen 
        
        # Now we can select the base data
        self.base_df = self.df_experian[self.df_experian["personid"].isin(valid_ids)].copy()
        dataInsight.unique_IDs(self.base_df,"base Experian dataset") # show number of IDs
        
        #------------------------ SAVE & RETURN -------------------------
        if self.save_intermediate:
            utils.save_df_to_csv(self.base_df, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")        
        #return base_df
        
        print("-------------------------------------")
        
        
        
    def create_base_cross_section(self,
                                  subsample = False,
                                  sample_size = 1000, 
                                  date_string = None,
                                  quarterly = True,
                                  outname = "cross_experian",
                                  seed = 1234):  
        """Method to create a cross-sectional dataset of the unique person ids.
        For this cross-sectional dataset we use only the information of the 
        customers who are a customer of Knab at the specified date. 
        We use their activity histories until the specified time and their
        characteristics at the specified time.
        Result: two datasets, one that the model is made on and then the next
        quarter or month which can be used to test predictions"""

        #TODO kan base_df een input van de functie maken
        print(f"****Creating cross-sectional data, at {utils.get_time()}****")
        
        #-------------Check that date is entered correctly--------------

        # First validate that the entered date is in the correct format
        time_check = "%Y-%m"
       
        if (date_string!=None): # If we take a date in the past for our cross-section
            try:
                date = datetime.strptime(date_string, time_check)
            except ValueError:
                print("This is the incorrect date string format. It should be YYYY-MM")
            
            # Also validate that it is not later than the most recent date in 
            # the data
            assert date <= self.last_date, \
            "This is later than the most recent available data"
        else:
            date = self.last_date

        time_format = "%Y-%m-%d"
        time_string = date.strftime(time_format)
       
        
        #-----------Taking a time-slice of Experian data for base dataframe-----------
        
        ## We want to work in either quarterly or monthly data. So for the
        ## specific date we want to take the last day of the month or the quarter
        if quarterly:
            print(f"Using quarterly data, time-slice date is {time_string}")
            quarter = ((date.month-1)//3)+1  #the quarter of our time slice
            print(f"quarter is Q{quarter}, we take the last date of the quarter")
  
            cross_date = self.get_last_day_period(date,next_period=False,quarterly=True)
            next_date = self.get_last_day_period(date,next_period=True,quarterly=True)
        
        else:
            print(f"Using monthly data, time-slice date is {time_string}")
            month = date.month
            print(f"month is {month}, we take the last date of the month")

            cross_date = self.get_last_day_period(date,next_period=False,quarterly=False)
            next_date = self.get_last_day_period(date,next_period=True,quarterly=False)
            
        print(f"cross-date {cross_date.strftime(time_format)},")
        print(f"next date {next_date.strftime(time_format)}")
        
        
        ## Now get the base dataset for this current period and the next 
        df_cross = self.get_time_slice(self.base_df,cross_date)
        df_next = self.get_time_slice(self.base_df,next_date)
        print(f"There is data for {len(df_cross)} customers at the cross-section point")
        #print(f"{len(df_next)}")
        
        
        #------------------------ SAVE & RETURN -------------------------
        
        if self.save_intermediate:
            utils.save_df_to_csv(df_cross, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")
        
        return df_cross, cross_date, df_next, next_date
           
        print("-------------------------------------")



    def create_cross_section_perportfolio(self, df_cross, cross_date, 
                                          outname = "df_cross_portfoliolink",
                                          quarterly=True): 
        """Creates a dataset of information for a set of people at a 
        specific time"""
        

        #--------------------- CREATE CROSS-SECTIONAL DATASET ---------------------
        
        #----------- Portfolio + business information (per portfolio) --------
        print(f"****Getting time-sliced linking data, at {utils.get_time()}****")
        
        # Start by taking a cross-section of the dataset linking 
        # portfolio ids to person ids and to business+portfolio information
        df_cross_link = self.get_time_slice(self.df_link,cross_date,
                                            valid_to_string = 'validtodate',
                                            valid_from_string = 'validfromdate')
        
        # Take the same subsample of people as what we took from the experian data
        df_cross_link = df_cross_link.loc[df_cross_link["personid"].isin(
            df_cross["personid"].unique())]
        
        # Get total portfolio ownership (including the ones we don't have info for)
        # Before we remove the portfolios without information
        frequencies = df_cross_link["personid"].value_counts().rename_axis('personid').to_frame('portofliocountstotal').reset_index(level=0)

    
        #----- Get only the ACTIVE portfolios (for which we have info) -----
        
        # get the unique portfolio ids which are still active at cross-section time
        df_portfolio_status = self.get_time_slice(pd.read_csv(f"{self.indir}/portfolio_status.csv"),
                                             cross_date,
                                             valid_to_string = 'outflow_date',
                                             valid_from_string = 'inflow_date')
        
        df_cross_link = df_cross_link.loc[df_cross_link["portfolioid"].isin(
            df_portfolio_status["portfolioid"].unique())]
        
        
        #-------------- Add Transaction information (per portfolio) --------------
        print(f"****Summarizing transaction activity per portfolio, at {utils.get_time()}****")
                
        temp_dataset = self.create_transaction_data_crosssection(cross_date)
        df_transactions = self.summarize_transactions(dataset = temp_dataset,
                                                      date = cross_date,
                                                      quarterly_period = quarterly)
        # utils.save_df_to_csv(df_transactions, self.outdir, 
        #                       "z_sumtransactions", add_time = False )    
        
        # Merge with the big linking dataset
        df_cross_link = df_cross_link.merge(df_transactions, 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 
        
        #-------------- Add activity information (per portfolio) -----------
        print(f"****Summarizing activity per portfolio, at {utils.get_time()}****")
                
        temp_dataset = self.create_activity_data_crosssection(cross_date)
        df_activity = self.summarize_activity(dataset = temp_dataset,
                                                    date = cross_date,
                                                    quarterly_period = quarterly)
        # utils.save_df_to_csv(df_activity_retail, self.outdir, 
        #                       "z_sumactivity", add_time = False )      
        
        # Merge with the big linking dataset
        df_cross_link = df_cross_link.merge(df_activity, 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 
        
        # Change status to numbers, to make it easier to work with
        df_cross_link.replace({'activitystatus': {'4_primaire bank':4,
                                                  '3_actief':3,
                                                  '2_SparenOnlyYN':2,
                                                  '1_inactief':1}},inplace=True)
        
        #-------------- Add bookkeeping overlay data (per portfolio) --------------
        print(f"****Summarizing bookkeeping overlay data per portfolio, at {utils.get_time()}****")
        
        df_portfolio_boekhoudkoppeling = self.get_time_slice(pd.read_csv(f"{self.indir}/portfolio_boekhoudkoppeling.csv"),
                                             cross_date,
                                             valid_to_string = 'valid_to_dateeow',
                                             valid_from_string = 'valid_from_dateeow')
       
        # We do not use some of these variables, they are not relevant
        df_portfolio_boekhoudkoppeling.drop(["boekhoudkoppeling","accountid",
                                             "valid_from_dateeow","valid_to_dateeow"],
                                            1,inplace=True)
        
        # Pak het aantal unieke overlay ids per combinatie van persoon en portfolio
        df_portfolio_boekhoudkoppeling.drop_duplicates(inplace=True)
        df_portfolio_boekhoudkoppeling = df_portfolio_boekhoudkoppeling.groupby(["personid","portfolioid"]).size().reset_index(name="accountoverlays")
        # utils.save_df_to_csv(df_portfolio_boekhoudkoppeling, self.outdir, 
        #                         "z_sumoverlays", add_time = False )  
        
        # merge with the portfolio link dataset
        df_cross_link = df_cross_link.merge(df_portfolio_boekhoudkoppeling,
                                            how="left", left_on=["personid","portfolioid"],
                                            right_on=["personid","portfolioid"])
        
        #------------------------ SAVE & RETURN -------------------------
        
        if self.save_intermediate:
            utils.save_df_to_csv(df_cross_link, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")

        return df_cross_link  
        print("-------------------------------------")
        
        
        
        
    def create_cross_section_perperson(self, df_cross, df_cross_link,
                                       cross_date, outname = "final_df",
                                       quarterly=True):  
        """This function takes a dataset linking each person ID to
        information for each portfolio separately, and returns a dataset
        where all the portfolio information is aggregated per person ID"""
       
        print(f"****Summarizing all information per ID, at {utils.get_time()}****")
        
        #-------------------------------------------------------------------
        
        # Create a dictionary of the data separated by portfolio type
        df_cross_link_dict = {}
        df_cross_link_dict['business'] = df_cross_link[df_cross_link["type"]=="Corporate Portfolio"]
        df_cross_link_dict['retail'] = df_cross_link[(df_cross_link["type"]=="Private Portfolio")\
                                             & (df_cross_link["enofyn"]==0)]
        df_cross_link_dict['joint'] = df_cross_link[(df_cross_link["type"]=="Private Portfolio")\
                                             & (df_cross_link["enofyn"]==1)]
        
            
        # For activity information, we will take the average activity PER portfolio type
        for name, df in df_cross_link_dict.items():
            
            
            #---------- variables for portfolio types ------------
            # create a column of ones and the person id for each portfolio type
            indicator = df.loc[:,["personid"]]
            indicator[name] = 1 # make all of the value one and give it the name
            indicator = indicator.groupby("personid").sum() # We want the amount of portfolios?
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            #---------- Incoporate the activity variables ------------
            indicator = df.loc[:,["personid"]]
            indicator[f"aantaltegenrekeningenlaatsteq_{name}"] = df.loc[:,"aantaltegenrekeningenlaatsteq"]
            indicator[f"aantalloginsapp_{name}"] = df.loc[:,"aantalloginsapp"]
            indicator[f"aantalloginsweb_{name}"] = df.loc[:,"aantalloginsweb"]
            indicator[f"activitystatus_{name}"] = df.loc[:,"activitystatus"]
            
            # We pakken het MAXIMUM om de meest actieve rekening weer te geven
            # alternatief: gemiddelde pakken?
            indicator = indicator.groupby("personid").max()
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            #---------- Incoporate the transaction variables ------------
           
            indicator = df.loc[:,["personid"]]
            # Take the variables for number of transactions
            indicator[f"aantalbetaaltransacties_{name}"] = df.loc[:,"aantalbetaaltransacties"]
            indicator[f"aantalatmtransacties_{name}"] = df.loc[:,"aantalatmtransacties"]
            indicator[f"aantalpostransacties_{name}"] = df.loc[:,"aantalpostransacties"]
            indicator[f"aantalfueltransacties_{name}"] = df.loc[:,"aantalfueltransacties"]
            
            # Take the variables indicating account ownership
            # We pakken voor deze binaire variabelen (depositoyn,etc) ook de 
            # MAXIMUM (dus dan is het 1 als 1 van de rekeningen het heeft)
            indicator[f"betalenyn_{name}"] = df.loc[:,"betalenyn"]
            indicator[f"depositoyn_{name}"] = df.loc[:,"depositoyn"]
            indicator[f"flexibelsparenyn_{name}"] = df.loc[:,"flexibelsparenyn"]
            indicator[f"kwartaalsparenyn_{name}"] = df.loc[:,"saldokwartaalsparen"]
            indicator[f"aantalfueltransacties_{name}"] = df.loc[:,"aantalfueltransacties"]
            
            # We pakken het MAXIMUM om de meest actieve rekening weer te geven
            indicator = indicator.groupby("personid").max()
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            
            # Maar voor de saldos pakken we de SOM over alles 
            indicator = df.loc[:,["personid"]]
            indicator[f"saldototaal_{name}"] = df.loc[:,"saldototaal"]
            
            indicator = indicator.groupby("personid").sum()
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            
            #------ Variables that are specific to portfoliotype -------
            if name == 'joint': 
                # Add gender for the joint portfolio?
                # Possibly do other things for the joint porfolio data to
                # take into account that people are more active?
                indicator = df.loc[:,["personid"]]
                indicator[f"geslacht_{name}"] = df.loc[:,"geslacht"]
                indicator = indicator.drop_duplicates(subset=["personid"])
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            
            if name == 'business':
                # See if they have a bookkeeping overlay
                indicator = df.loc[:,["personid"]][df["accountoverlays"]>0] 
                indicator["accountoverlay"]=1 # make all of the columns one and give it the name
                # now sum to find for how many business portfolios there is an 
                # account overlay
                indicator = indicator.groupby("personid").sum() 
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
                
                # TODO make some kind of summary for the business details
                # -> voor bedrijf types pakken we de type OF we pakken ''meerdere''
                #indicator = df.loc[:,["personid", "birthday", "subtype", "code", "name"]]
                # Dit geeft een error!! birthday, subtype etc is missing??
                
                # doe een merge van de indicator met df_cross[name] oftewel 
                # df_cross['business'] wat aangeeft of er meerdere 
                # bedrijfsportfolios zijn voor die persoon?
                
                
                # We laten duplicates alvast vallen
                indicator = indicator.drop_duplicates()
                
                # TODO: fill in the blank spaces who are missing with 0??
                # Need to differentiate between missing data and data that is
                # not there?
            
            
        #----------- Merge remaining things --------
        
        # TODO check if this is necessary? And if so, make it so that 
        # frequencies is an input to the function since it is created in a different function
        # merge frequencies of ALL portfolios (including ones without any info)
        # which we made earlier 
        # df_cross= df_cross.merge(frequencies,
        #                           how="left", left_on=["personid"],
        #                           right_on=["personid"],)
        
        
        # Get the personid and birthyear which were stored in crosslink 
        # - first drop the duplicates
        characteristics = df_cross_link.loc[:,["personid","birthyear","geslacht",
                "enofyn"]]
        
        # Sort the characteristics by enofyn, so that the joint portfolios appear later.
        # then drop duplicates and keep the first entries that appear.
        characteristics = characteristics.sort_values("enofyn")
        characteristics = characteristics.drop_duplicates(subset=["personid"],
                          keep='first', inplace=False)                               
        df_cross= df_cross.merge(characteristics, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],) 
        
        
        #TODO: zorg dat de pakketcategorie en een paar 'algemene' variabelen
        # er nog in komen??
        
        #------------------------ SAVE & RETURN -------------------------
        
        #if self.save_intermediate:
        utils.save_df_to_csv(df_cross, self.interdir, 
                             outname, add_time = False )      
        print(f"Finished and output saved, at {utils.get_time()}")

       #return df_cross
           
        print("-------------------------------------")
        


#some helper methods that are used to handle time =================================


    def get_last_day_period(self, date, next_period = False, quarterly =True):
        """Returns the last day of the period that contains our date, by getting 
        the day BEFORE the FIRST day of the next period """
        
        if quarterly:
            quarter = ((date.month-1)//3)+1  #the quarter of our time slice            
            if (quarter<4):
                if (next_period):
                    end_date = datetime(date.year, (3*(quarter+1))%12 +1, 1) + timedelta(days=-1)
                else:
                    end_date = datetime(date.year, (3*quarter)%12 +1, 1) + timedelta(days=-1)
            else: # next period crosses onto the next year
                if (next_period):
                    end_date = datetime(date.year+1, (3*(quarter+1))%12 +1, 1) + timedelta(days=-1)
                else:
                    end_date = datetime(date.year+1, (3*quarter)%12 +1, 1) + timedelta(days=-1)
        
        else: # We work in periods of months
            month = date.month
            if (month<12):
                if (next_period):
                    end_date = datetime(date.year, (month+1)%12 +1, 1) + timedelta(days=-1)
                else:
                    end_date= datetime(date.year, (month)%12 +1, 1) + timedelta(days=-1)
            else:
                if (next_period):
                    end_date = datetime(date.year+1, (month+1)%12 +1, 1) + timedelta(days=-1)
                else:
                    end_date = datetime(date.year+1, (month)%12 +1, 1) + timedelta(days=-1)
                    
        return end_date
        
    
    
    def get_time_slice(self, dataset, date, 
                       valid_to_string = 'valid_to_dateeow',
                       valid_from_string = 'valid_from_dateeow'):
        """Method to return a slice of a dataset based on a certain date"""
        
        # Make sure the date variables are in datetime format
        # The errors=coerce changes the out of bounds dates (9999-12-31) to NaT
        dataset.loc[:,valid_to_string] = pd.to_datetime(dataset[valid_to_string], errors='coerce')
        dataset.loc[:,valid_from_string] = pd.to_datetime(dataset[valid_from_string], errors='coerce')    
        
        ## Now select those customer information points which were valid
        ## at the specified date
        select = ((dataset[valid_to_string]>= date) | dataset[valid_to_string].isnull()) \
            & (dataset[valid_from_string]<= date) 
        dataset = dataset[select]
        return dataset
    
    
    def select_period(self, dataset, last_day, quarterly_period):
        """Provides a vector of true/false values for selecting only the
        values within a specific period from the dataset"""
        if quarterly_period:
             # Select only the ones that are in the current quarter
             current_quarter = ((last_day.month-1)//3)+1

             # We already have filtered out everything after the last day of
             # the quarter, now we filter out everything before
             initial_day = datetime(last_day.year, 3 * current_quarter - 2, 1) 
             select = (dataset["dateeow"] >= initial_day)
             
        else:
             # We already have filtered out everything after the last day of
             # the month, now we filter out everything before
             initial_day = datetime(last_day.year, last_day.month, 1) 
             select = (dataset["dateeow"] >= initial_day)
             
        return select
    

    
#methods to process corporate data ===================================================
    
    def processCorporateData(self):
        """processes the corporate information data and creates some extra
        variables"""
        
        #-------------READ IN CORPORATE DETAILS AND PROCESS------------
        self.df_corporate_details = pd.read_csv(f"{self.indir}/corporate_details.csv")

        if self.print_info: # Print NaNs and most common values
            nameList = ["personid", "subtype", "name"]
            nameList2 = ["personid", "birthday", "subtype", "code", "name"]
            print("unique number of businessID's in corporate data :",self.df_corporate_details["subtype"].unique().shape) 
            dataInsight.numberOfNaN(self.df_corporate_details, nameList2)
            dataInsight.mostCommonDict(self.df_corporate_details, nameList, 10)

        # Copy DataFrame for editing and drop code column and rows with NaN value
        self.df_corporate_details = self.df_corporate_details.copy()
        self.df_corporate_details.dropna(inplace=True)

        # rename column personid to companyID,name to companySector and 
        tempDict = {
            "subtype": "businessType",
            "name": "businessSector"}
        self.df_corporate_details.rename(columns=tempDict, inplace=True)

        if self.print_info: # show dimensions
            print("shape of current data",self.df_corporate_details)

        #---------- CREATE foundingDate, companyAgeInDays AND foundingYear--------

        # use shorter var to refer to new columns
        aid = "businessAgeInDays"
        aim = "businessAgeInMonths"
        aiy = "businessAgeInYears"
        foundingDateString = "foundingDate"

        # convert to datetime and to age in days of company
        self.df_corporate_details[foundingDateString] = pd.to_datetime(self.df_corporate_details["birthday"])
        currentTime = datetime(2021, 1, 1)
        self.df_corporate_details["timeDelta"] = (currentTime - self.df_corporate_details[foundingDateString])
        self.df_corporate_details[aid] = self.df_corporate_details["timeDelta"].dt.days
        self.df_corporate_details[aim] = self.df_corporate_details["timeDelta"] / np.timedelta64(1, "M")
        self.df_corporate_details[aiy] = self.df_corporate_details["timeDelta"] / np.timedelta64(1, "Y")
        self.df_corporate_details.drop("timeDelta", inplace=True, axis=1)

        # note founding year of company
        self.df_corporate_details["foundingYear"] = self.df_corporate_details[foundingDateString].dt.year

        # assumption that a company date beyond the end of 2020 is faulty
        self.df_corporate_details = self.df_corporate_details[self.df_corporate_details[aid] > 0].copy()
        
        if self.print_info:
            print(self.df_corporate_details["birthday"].describe())
            
            dataInsight.mostCommon(self.df_corporate_details, aid, 10)
            print(self.df_corporate_details[aid].describe())
            self.df_corporate_details.sample(5)
    
            dataInsight.mostCommonDict(self.df_corporate_details, ["businessType", "foundingYear"], 10)
            self.df_corporate_details[aid].describe()

        a = self.df_corporate_details.index[:10].to_list()
        a.append(220682)
        self.df_corporate_details[self.df_corporate_details.index.isin(a)]

        #-------------PROCESS SBI CODES------------
        #TODO: add comments to describe what is happening here
        SBI_2019Data = pd.read_excel("SBI_2019.xlsx")

        tempString = "SBIcode"
        tempString2 = "SBIname"

        SBI_2019DataEdited = SBI_2019Data.copy()
        SBI_2019DataEdited.rename(columns={
            "Unnamed: 0": tempString,
            "Standaard Bedrijfsindeling 2008 - update 2019 ":
                tempString2}, inplace=True)
        codesList = SBI_2019DataEdited[tempString].unique().tolist()

        sectorList = []
        tempList = ["A", 0, 0]
        tempList2 = ["A"]
        for value in codesList[1:]:
            try:
                tempList[2] = int(value)
            except:
                pass
            try:
                text = str(value)
                text = text.replace("\xa0", "")
                text = text.replace(" ", "")
                val = ord(text)
                if 64 < val < 123:
                    sectorList.append(tempList.copy())
                    tempList2.append(text)
                    tempList[0] = text
                    tempList[1] = tempList[2]
                else:
                    pass
            except:
                pass
        sectorList.append(tempList.copy())
        sectorData = pd.DataFrame(tempList2)
        sectorData.rename({
                              0: tempString}, axis=1, inplace=True)
        sectorData = pd.merge(SBI_2019DataEdited[[tempString, tempString2]], sectorData, how="inner", on=tempString)
        sectorData.loc[13, [tempString, tempString2]] = "U", "Extraterritoriale organisaties en lichamen"
        sectorData.rename({
                              tempString: "SBIsector",
                              tempString2: "SBIsectorName"}, axis=1, inplace=True)

        SBI_2019DataEdited = SBI_2019DataEdited.loc[:, [tempString, tempString2]]
        SBI_2019DataEdited.dropna(inplace=True)
        SBI_2019DataEdited.reset_index(drop=True, inplace=True)

        def cleanSBI(x):
            try:
                return int(x)
            except:
                return np.nan

        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].apply(lambda x: cleanSBI(x))
        SBI_2019DataEdited.dropna(inplace=True)
        SBI_2019DataEdited[tempString2] = SBI_2019DataEdited[tempString2].astype("str")

        SBI_2019DataEdited["SBIsector"] = np.nan
        # chr, ord, < <=
        for values in sectorList:
            tempIndex = (values[1] < SBI_2019DataEdited[tempString]) & (SBI_2019DataEdited[tempString] <= values[2])
            SBI_2019DataEdited.loc[tempIndex, "SBIsector"] = values[0]

        SBI_2019DataEdited = pd.merge(SBI_2019DataEdited, sectorData, how="inner", on="SBIsector")

        tempString = "SBIcode"
        self.df_corporate_details[tempString] = self.df_corporate_details["businessSector"].str[:2]
        self.df_corporate_details[tempString] = pd.to_numeric(self.df_corporate_details[tempString],
                                                                downcast="unsigned")

        self.df_corporate_details = pd.merge(self.df_corporate_details, SBI_2019DataEdited, how="inner",
                                               on="SBIcode")
        self.df_corporate_details.drop(["code", "businessSector", "birthday"], axis=1, inplace=True)
    
    
#methods to create transction & activity data ============================================
    def create_transaction_data_crosssection(self, date = None):
        """creates transaction data based on the cross-section for a specific date"""
        
        #Deze variabelen zijn in sommige jaren altijd 0 dus gebruiken we niet
        # cols_to_drop = ["yearweek", "gemaksbeleggenyn", "saldogemaksbeleggen", "participatieyn",
        #               "saldoparticipatie","vermogensbeheeryn", "saldovermogensbeheer"]
        readArgs = {"usecols": declarationsFile.getPatColToParseCross(subset = "transactions")}

        if date.year >2019:
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_transaction_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_retail_2020.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transaction_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_business_2020.csv", **readArgs)]
            , ignore_index= True)

        elif date.year == 2019:     
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_transaction_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_retail_2019.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transaction_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_business_2019.csv", **readArgs)]
            , ignore_index= True)
        elif date.year == 2018:
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_transaction_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_retail_2018.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transaction_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transactions_business_2018.csv", **readArgs)]
            , ignore_index= True
            )
        else: #  year < 2018
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_transaction_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_transaction_business.csv", **readArgs)]
            , ignore_index= True
            )
            
        print(f"done concatenating transactions data at {utils.get_time()}")
    
        # Make sure the date variable is in datetime format
        dataset.loc[:,"dateeow"] = pd.to_datetime(dataset["dateeow"])
        
        # The transaction entries after the cross-section date are not useful
        dataset = dataset[dataset["dateeow"]<= date]
                
        return dataset

    def create_activity_data_crosssection(self, date):
        """creates activity data based on the cross-section for a specific date"""
        
        # Get the columns that we use - we do not need the payment alert variables
        readArgs = {"usecols": declarationsFile.getPatColToParseCross(subset = "activity")}
        
        # Read the necessary data
        if date.year >2019:
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_retail_2020.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business_2020.csv", **readArgs)]
            , ignore_index=True
            )
        elif date.year == 2019:     
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_retail_2019.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business_2019.csv", **readArgs)]
                , ignore_index=True
            )
        elif date.year == 2018:
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_retail_2018.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business_2018.csv", **readArgs)]
            , ignore_index= True
            )
        else: #  year < 2018
            dataset = pd.concat(
            [pd.read_csv(f"{self.indir}/portfolio_activity_retail.csv", **readArgs),
            pd.read_csv(f"{self.indir}/portfolio_activity_business.csv", **readArgs)]
            , ignore_index = True
            )
            
        print(f"done concatenating activity data at {utils.get_time()}")
        
        # Make sure the date variable is in datetime format
        dataset.loc[:,"dateeow"] = pd.to_datetime(dataset["dateeow"])
        
        # The activity entries after the cross-section date are not useful
        dataset = dataset[dataset["dateeow"]<= date]
                
        return dataset


    def summarize_activity(self, dataset, date, quarterly_period):
        """We summarize the activity per period, taking for logins the sum of
        all values in that quarter and for the remaining variables we take 
        the last entry for that value in the period"""
        
        select = self.select_period(dataset, date, quarterly_period)
         
        # We sum the login information, only for this time period. 
        logins = dataset[["portfolioid", "aantalloginsapp","aantalloginsweb" ]][select]
        logins = logins.groupby("portfolioid").sum()
    
        # For the remaining variables, we want the entry that is the closest
        # to the current date        
        remain = dataset.drop(["aantalloginsapp","aantalloginsweb" ], 1)
        maxdates = remain[["dateeow","portfolioid"]].groupby("portfolioid").max()
        # Do a left join on maxdates so that only the most recent entries are left
        remain = maxdates.merge(remain, how="left", left_on=["dateeow","portfolioid"],
                               right_on=["dateeow","portfolioid"])
        # merge them again
        dataset = logins.merge(remain, how="left", left_on=["portfolioid"],
                               right_on=["portfolioid"],)

        #TODO grotere kans op missing values als er geen voorgaande data is??
        # Oplossing: sowieso alleen naar de afgelopen maand/ etc kijken, anders 
        # krijgt het eventueel iets als inactive ofzo?
        #TODO: maak een variabele van aantal weken sinds laatste dateeow?
                
        dataset = dataset.rename(columns={"dateeow": "lastactivityeow",})
        return dataset


    def summarize_transactions(self, dataset, date, quarterly_period):
        """We summarize the transactions per period, taking for logins the sum of
        all values in that quarter and for the remaining variables we take 
        the last entry for that value in the period"""

        select = self.select_period(dataset, date, quarterly_period)
        
        transactions = dataset[["portfolioid", "aantalbetaaltransacties", 
                          "aantalatmtransacties","aantalpostransacties",
                          "aantalfueltransacties" ]][select]
        transactions = transactions.groupby("portfolioid").sum()
    
        # For the remaining variables, we want the record that is the closest
        # to the current date        
        remain = dataset.drop(["aantalbetaaltransacties","aantalatmtransacties",
                               "aantalpostransacties","aantalfueltransacties" ], 1)
        maxdates = remain[["dateeow","portfolioid"]].groupby("portfolioid").max()
        # Do a left join on maxdates so that only the most recent entries are left
        remain = maxdates.merge(remain, how="left", left_on=["dateeow","portfolioid"],
                               right_on=["dateeow","portfolioid"])
        # merge them again
        dataset = transactions.merge(remain, how="left", left_on=["portfolioid"],
                               right_on=["portfolioid"],)
        
        dataset = dataset.rename(columns={"dateeow": "lasttransaceow",})
        
        return dataset


    def importPortfolioActivity(self, convertData=False, selectColumns=False, 
                                discardPat = False, **readArgs):

        mergeSet = ["dateeow", "yearweek", "portfolioid", "pakketcategorie"]

        if selectColumns:
            readArgsAct = {**readArgs,
                        "usecols": declarationsFile.getPatColToParseTS("activity")}
            readArgsTrans = {**readArgs,
                        "usecols": declarationsFile.getPatColToParseTS("transaction")}
            readArgs = {**readArgsTrans, **readArgsAct}
            mergeSetNew = []
            for item in mergeSet:
                if item in readArgs["usecols"]:
                    mergeSetNew.append(item)
            mergeSet = mergeSetNew
        else:
            readArgsAct = {**readArgs}
            readArgsTrans = {**readArgs}

        tempList = [f"{self.indir}/portfolio_activity_business_2018.csv", f"{self.indir}/portfolio_activity_business_2019.csv",
                    f"{self.indir}/portfolio_activity_business_2020.csv"]
        pab1820 = utils.importAndConcat(tempList, **readArgsAct)

        print(pab1820.shape, " are the dimensions of pab18-20")

        tempList = [f"{self.indir}/portfolio_activity_retail_2018.csv", f"{self.indir}/portfolio_activity_retail_2019.csv",
                    f"{self.indir}/portfolio_activity_retail_2020.csv"]
        par1820 = utils.importAndConcat(tempList, **readArgsAct)
        print(par1820.shape, " are the dimensions of par18-20")

        pa1820 = pd.concat(
            [pab1820, par1820], ignore_index=True)
        del par1820, pab1820
        gc.collect()
        print(pa1820.shape, " are the dimensions of pa 18-20 before merge")
        if convertData:
            pa1820 = utils.doConvertFromDict(pa1820)

        tempList = [f"{self.indir}/portfolio_activity_transactions_business_2018.csv",
                    f"{self.indir}/portfolio_activity_transactions_business_2019.csv",
                    f"{self.indir}/portfolio_activity_transactions_business_2020.csv"]
        patb1820 = utils.importAndConcat(tempList, **readArgsTrans)
        print(patb1820.shape, " are the dimensions of patb 18-20")

        tempList = [f"{self.indir}/portfolio_activity_transactions_retail_2018.csv",
                    f"{self.indir}/portfolio_activity_transactions_retail_2019.csv",
                    f"{self.indir}/portfolio_activity_transactions_retail_2020.csv"]
        patr1820 = utils.importAndConcat(tempList, **readArgsTrans)
        print(patr1820.shape, " are the dimensions of patr 18-20")

        pat1820 = pd.concat(
            [patr1820, patb1820],
            ignore_index=True)
        print(pat1820.shape, " are the dimensions of pa before merge 18-20")
        del patr1820, patb1820
        gc.collect()
        if convertData:
            pat1820 = utils.doConvertFromDict(pat1820)

        pat1820 = pd.merge(pa1820,
                           pat1820, how="inner",
                           on=mergeSet)
        del pa1820
        gc.collect()
        print(pat1820.shape, " are the dimensions of pat 18-20")

        if convertData:
            pat1820 = utils.doConvertFromDict(pat1820)

        # Todo verander of dee naam van deze bestanden of de naam van de andere bestanden
        tempList = [f"{self.indir}/portfolio_activity_business.csv", f"{self.indir}/portfolio_activity_retail.csv", ]
        pa1420 = utils.importAndConcat(tempList,chunkSize = 250000, **readArgsAct)
        print(pa1420.shape, " are the dimensions of pa before merge 14-20")

        tempList = [f"{self.indir}/portfolio_activity_transaction_business.csv", f"{self.indir}/portfolio_activity_transaction_retail.csv"]
        pat1420 = utils.importAndConcat(tempList, chunkSize = 250000,**readArgsTrans)
        print(pat1420.shape, " are the dimensions of pa before merge 14-20")
        patotal1420 = pd.merge(pa1420,
                               pat1420, how="inner",
                               on=mergeSet)
        del pa1420, pat1420
        gc.collect()
        print(patotal1420.shape, " are the dimensions of pat 14-20")

        if convertData:
            patotal1420 = utils.doConvertFromDict(patotal1420)

        pat = pd.concat([patotal1420, pat1820])
        print(pat.shape, " are the dimensions of pat 14-20")

        self.df_pat = pat

    def portfolioActivitySampler(self, n = 4000, export_after = False, replaceGlobal = False):
        randomizer = np.random.RandomState(self.seed)
        if self.df_pat.empty:
            self.importPortfolioActivity()
        uniqueList = self.df_pat['portfolioid'].unique()
        chosenID = randomizer.choice(uniqueList, n)
        indexID = self.df_pat['portfolioid'].isin(chosenID)
        self.df_pat_sample = self.df_pat[indexID].copy()
        if replaceGlobal:
            self.df_pat = self.df_pat_sample.copy()
        self.exportEdited("patsmp")


#linkTimeSets=====================================================================

    def linkTimeSets(self, period = "Q"):
        # ToDo corrigeer voor al geimporteerde of bewerkte data
        # Todo zorg er voor dat valid to date wordt gepakt
        # TODO kleine sample nemen voor data
        #TODO koppel portfolioinfo
        #TODO koppel corporate
        print(f"****linking timeseries sets, starting at {utils.get_time()}****")
        # If valid_to < period in Time Series
        # if self.df_expTS.empty:
        #     self.transformExperianTS(period = period)
        if self.df_corporate_details.empty:
            self.processCorporateData()

        if self.df_pat.empty:
            self.importPortfolioActivity(convertData= True, selectColumns= True)

        df_lpp = pd.read_csv(f'{self.indir}/linkpersonportfolio.csv')
        df_exp = pd.read_csv(f'{self.indir}/experian.csv')
        df_bhk = pd.read_csv(f'{self.indir}/portfolio_boekhoudkoppeling.csv')
        df_pin = pd.read_csv(f'{self.indir}/portfolio_info.csv')
        df_cor = self.df_corporate_details


        #TO CREATE QUICKER SAMPLE
        pat_unique = self.df_pat['portfolioid'].unique()
        #SAMPLEFORTEST
        df_lpp = df_lpp.loc[df_lpp['portfolioid'].isin(pat_unique),:]
        #SAMPLEFORTESTEND

        df_lpp = utils.doConvertFromDict(df_lpp)
        df_lpp_table_port_corp_retail = df_lpp.groupby("portfolioid")["iscorporatepersonyn"].mean()
        df_lpp_table_port_corp_retail = df_lpp_table_port_corp_retail.reset_index()
        df_lpp_table_port_corp_retail.loc[:,"indicator_corp_and_retail"] = 0
        lpp_index_both = pd.eval(" (df_lpp_table_port_corp_retail['iscorporatepersonyn'] > 0) & "
                                 "(df_lpp_table_port_corp_retail['iscorporatepersonyn'] < 1) ")
        df_lpp_table_port_corp_retail.loc[lpp_index_both,"indicator_corp_and_retail"] = 1
        df_lpp_table_port_corp_retail.drop("iscorporatepersonyn", axis=1, inplace=True)
        df_lpp.drop(["validfromdate", "validfromyearweek", "validtodate"], axis=1, inplace=True)
        df_lpp = pd.merge(df_lpp,df_lpp_table_port_corp_retail, on = "portfolioid")

        index_link_cr = pd.eval("df_lpp['indicator_corp_and_retail'] == 1")
        index_corp_pers = pd.eval("df_lpp['iscorporatepersonyn'] == 1")
        portid_link_cr = df_lpp.loc[index_link_cr, "portfolioid"].unique()
        persid_link_cr = df_lpp.loc[index_link_cr, "personid"].unique()
        persid_no_link_cr = df_lpp.loc[~index_link_cr, "personid"].unique()
        persid_no_link_cr = persid_no_link_cr[~persid_no_link_cr.isin(persid_link_cr)]
        portid_no_link_cr = df_lpp.loc[~index_link_cr, "portfolioid"].unique()
        port_no_link_cr = portid_no_link_cr[~portid_no_link_cr.isin(portid_link_cr)]

        # id_link_cr_corp = df_lpp.loc[(index_link_cr & index_corp_pers) , "personid"].unique()
        # id_link_cr_ret = df_lpp.loc[(index_link_cr & ~index_corp_pers), "personid"].unique()
        # id_no_link_cr_corp = df_lpp.loc[(~index_link_cr & index_corp_pers), "personid"].unique()
        # id_no_link_cr_ret = df_lpp.loc[(~index_link_cr & ~index_corp_pers), "personid"].unique()

        # df_lpp_linked = df_lpp[pd.eval("df_lpp['personid'].isin(persid_link_cr)")]
        # df_lpp_only_corp = df_lpp[pd.eval("df_lpp['personid'].isin(id_no_link_cr_corp)")]
        # df_lpp_only_ret = df_lpp[pd.eval("df_lpp['personid'].isin(id_no_link_cr_ret)")]

        ### LINK LARGE SETS BOTH
        # pat_to_edit = pd.merge(self.df_pat, df_lpp, on="portfolioid")
        # jc1, jc2 = pat_to_edit.columns.get_loc("pakketcategorie"), pat_to_edit.columns.get_loc("personid")
        # pat_to_edit = pat_to_edit[["dateeow","personid", "portfolioid"] + pat_to_edit.columns[jc1:jc2].tolist() + pat_to_edit.columns[(jc2+1):].tolist() ]

        sampleCol = ["dateeow","portfolioid", "pakketcategorie", "activitystatus", "saldobetalen"]
        pat_to_edit = self.df_pat[sampleCol].copy()
        pat_to_edit = utils.doConvertFromDict(pat_to_edit)
        pat_to_edit.loc[pat_to_edit["dateeow"] == "2021-01-03 00:00:00","dateeow" ] = self.endDate

        #Link with PIN
        #TODO
        #Methods used to look into the data
        ztest3a1 = "dataInsight.mostCommon(df_pin['enofyn'] == 0], 'geslacht',5)"
        ztest3a2 = "dataInsight.mostCommon(df_pin['enofyn'] == 1], 'geslacht',5)"
        ztest3a3 = """only_man_female_pin = df_pin[pd.eval("df_pin['geslacht'] != 'Man' & df_pin['geslacht'] != 'Vrouw'")]"""
        ztest3a4 = "(df_pin['portfolioid'].value_counts()>1).sum()"
        ztest3a5 = "pat_pin_dates_merge = pat_to_edit.groupby('portfolioid')['dateeow'].min()"
        ztest3a6 = "pat_pin_dates_merge = pd.merge(pat_pin_dates_merge, df_pin[['portfolioid','dateinstroomweek']], on = 'portfolioid')"

        ##IF SAMPLE##
        df_pin = df_pin[pd.eval("df_pin['portfolioid'].isin(pat_unique)")]
        ##END SAMPLING

        print(pat_to_edit.shape)

        joined_linked = pat_to_edit.copy()
        df_pin = utils.doConvertFromDict(df_pin)

        selected_columns_pin = ['portfolioid','dateinstroomweek', 'birthyear', 'geslacht', 'type','enofyn']
        joined_linked = pd.merge(joined_linked,df_pin[selected_columns_pin], on = 'portfolioid')#Are a few NA rows which start on the last date,
        # discarded for now

        joined_linked_port_start = joined_linked.groupby('portfolioid').aggregate({'dateeow':np.min, 'dateinstroomweek': np.min})
        joined_linked_port_start.reset_index(inplace= True)
        joined_linked_port_start.dropna(inplace=True)
        joined_linked_port_start.query("dateeow <= dateinstroomweek", inplace = True)



        #TODO check if na rows from df_pin need to be taken too or can be discarded

        exp2 = df_exp.copy()
        #SAMPLEFORTEST START
        exp2 = exp2[(exp2['personid'].isin(persid_link_cr)) | (exp2['personid'].isin(persid_no_link_cr))]
        #END SAMPLE

        #CHECK IF VALUES CAN BOTH HAVE INFO
        exp2['valid_to_dateeow'] = pd.to_datetime(exp2['valid_to_dateeow'])
        exp2['valid_to_dateeow'].fillna(self.endDate, inplace = True)
        exp2.sort_values(['valid_to_dateeow', 'personid'], inplace=True)
        exp2.dropna(axis = 0, subset = exp2.columns[3:].tolist(), how = 'all', inplace = True)
        index_only_buss_finergy = exp2['age_hh'].isna()
        # exp_only_buss_finergy = exp2[index_only_buss_finergy].copy()
        exp2 = exp2[~index_only_buss_finergy]
        exp2 = utils.doConvertFromDict(exp2)

        exp2['retail_id_with_corp_and_retail'] = 0
        exp2.loc[exp2['personid'].isin(persid_link_cr),'retail_id_with_corp_and_retail'] = 1

        #used later in merging
        value_counts_exp = exp2["personid"].value_counts()
        value_counts_multiple_exp = value_counts_exp[value_counts_exp > 2].reset_index()["index"]

        #Link pat_to_edit and
        temp_list = ['personid', 'portfolioid','indicator_corp_and_retail',"iscorporatepersonyn"]
        exp_lpp_joined = pd.merge(exp2, df_lpp[temp_list], on=['personid'])
        exp_lpp_joined.sort_values(['valid_to_dateeow','personid','portfolioid'], inplace= True)
        pat_to_edit.sort_values(['dateeow', 'portfolioid'], inplace=True)


        #Merge joined by picking records to match that are before valid_to_dateeow, not matching records or expired records discarded
        joined_linked = pd.merge(pat_to_edit, exp_lpp_joined, on="portfolioid")
        temp_index = pd.eval("(joined_linked['valid_from_dateeow'] <= joined_linked['dateeow']) & \
                            (joined_linked['valid_to_dateeow'] >= joined_linked['dateeow'])")
        joined_linked = joined_linked[temp_index]

        temp_index = pd.eval("pat_to_edit['portfolioid'].isin(joined_linked['portfolioid'])")
        joined_linked_no_exp = pat_to_edit[~temp_index]
        joined_linked = pd.concat([joined_linked, joined_linked_no_exp], ignore_index= True)


        #TODO Check if every associated business is coupled through portfolio.
        #TODO Can see how much personid's from corporate can be associated : Multiple merge, portfolio to person, then new person to portfolio

        # MERGE CORP WITH LPP
        corporate_columns_to_use = ['personid', 'businessType', 'businessAgeInDays', 'foundingYear', 'SBIcode', 'SBIname', 'SBIsector',
                                    'SBIsectorName']

        # SAMPLED SIZE#
        print("amount of observations of corp in lpp linked :", df_cor['personid'].isin(persid_link_cr).sum())

        df_cor = df_cor[corporate_columns_to_use]
        df_cor = utils.doConvertFromDict(df_cor)
        print("Dimension corporate details before merge :", df_cor.shape)


        temp_list = ['personid', 'portfolioid', 'indicator_corp_and_retail', "iscorporatepersonyn"]
        cor_lpp_linked = pd.merge(df_cor, df_lpp[temp_list], on="personid")
        print("Dimension of merged file :", cor_lpp_linked.shape)

<<<<<<< Updated upstream
        cor_lpp_linked['business_id_with_corp_and_retail'] = 0
        cor_lpp_linked.loc[cor_lpp_linked['personid'].isin(persid_link_cr), 'business_id_with_corp_and_retail'] = 1
=======
        #Todo verander naar een left of right join
>>>>>>> Stashed changes
        # Merge corp_lpp with large joined table

        cor_lpp_linked.rename({'personid':'businessid'},axis = 1)
        print(f"before merge dimension of joined_linked : {joined_linked.shape} and dimension of cor_lpp : {cor_lpp_linked.shape}")
        joined_linked = pd.merge(joined_linked, cor_lpp_linked, how = "left", on = "portfolioid", suffixes = ['','_business'])
        print(f"after merge dimension: {joined_linked.shape}")


        """
        Merge Boekhoudkoppeling to large file
        """
        ##SAMPLESTART
        df_bhk = df_bhk[df_bhk['portfolioid'].isin(pat_unique)]
        sample_columns = ['dateeow', 'personid', 'portfolioid', 'saldobetalen', 'age_hh', 'finergy_tp', 'businessid',
                          'businessAgeInDays', 'SBIsectorName']
        joined_linked = joined_linked[sample_columns]
        ##SAMPLEEND

        df_bhk = utils.doConvertFromDict(df_bhk)
        df_bhk.loc[df_bhk['valid_to_dateeow'].isna(), 'valid_to_dateeow'] = self.endDate
        df_bhk.drop(['accountid','accountoverlayid'], axis = 1, inplace = True) #Drop unused vars
        df_bhk.drop_duplicates(inplace = True)
        df_bhk.sort_values(['valid_to_dateeow', 'personid', 'portfolioid'], inplace=True)
        joined_linked.sort_values(['dateeow','personid','portfolioid'])

        #Chosen to merge on person rather than portfolio
        person_id_in_bhk = df_bhk['personid'].unique()
        before_merge_index_bhk = joined_linked['personid'].isin(person_id_in_bhk)


        #BHK Tests
        ztest11 =  "n1 = df_bhk.groupby('personid')[['personid','portfolioid', 'boekhoudkoppeling', 'valid_from_dateeow', 'valid_to_dateeow']]. \
            filter(lambda x: x['boekhoudkoppeling'].unique().shape[0] > 1) "
        ztest11a =  "n1.drop_duplicates(inplace=True)"
        ztest12 =  "n2 = df_bhk.groupby('portfolioid')[['personid','portfolioid', 'boekhoudkoppeling', 'valid_from_dateeow', 'valid_to_dateeow']]. \
            filter(lambda x: x['boekhoudkoppeling'].unique().shape[0] > 1) "
        ztest12a =  "n2.drop_duplicates(inplace=True)"
        ztest13 = "bhk_multiple_list = n1['personid'].unique()"
        ztest13a = "bhk_multiple_list = bhk_multiple_list[pd.Series(bhk_multiple_list).isin(joined_linked['personid'])]"
        ztest14 = "gen1, gen2, gen3, gen4, gen5, gen6 = dataInsight.checkAVL(bhk_multiple_list), dataInsight.checkAVL(bhk_multiple_list), \
                                                 dataInsight.checkAVL(bhk_multiple_list), dataInsight.checkAVL(bhk_multiple_list),\
                                                 dataInsight.checkAVL(bhk_multiple_list), dataInsight.checkAVL(bhk_multiple_list)"

        ztest15 = "ex1, ex2, ex3, ex4, ex5, ex6 = dataInsight.checkV1(joined_linked1, next(gen1)), dataInsight.checkV1(joined_linked2," \
                  " next(gen2), 'personid_x'), dataInsight.checkV1(joined_linked3, next(gen3)) ,dataInsight.checkV1(df_bhk, next(gen4)), \
                                       dataInsight.checkV1(exp_lpp_joined, next(gen5)),\
                                       dataInsight.checkV1(joined_linked, next(gen6))"
        ztest1all = "exec(ztest11),exec(ztest11a),exec(ztest13),exec(ztest13a),exec(ztest14),exec(ztest15)"

        print("with bhk dimensions before merge :",joined_linked.shape)



        templist = ['dateeow','personid','portfolioid']
        joined_linked_bkh = pd.merge(joined_linked.loc[before_merge_index_bhk, templist], df_bhk, on=['personid', 'portfolioid'])
        tempindex = pd.eval("(joined_linked_bkh['valid_from_dateeow'] <= joined_linked_bkh['dateeow']) & \
                                                (joined_linked_bkh['valid_to_dateeow'] >= joined_linked_bkh['dateeow'])")
        joined_linked_bkh = joined_linked_bkh[tempindex]
        # Probably erronous that bhk can be active one day only
        joined_linked_bkh = joined_linked_bkh[pd.eval("joined_linked_bkh['valid_from_dateeow'] != joined_linked_bkh['valid_to_dateeow']")].copy()
        print('Size file after merge :', joined_linked_bkh.shape)
        print('Not NA after merge :', joined_linked_bkh["boekhoudkoppeling"].notna().sum())
        print('Not NA after merge :', joined_linked_bkh["valid_from_dateeow"].notna().sum())

        joined_linked_bkh.drop(['valid_from_dateeow','valid_to_dateeow'],axis = 1, inplace = True)
        joined_linked = pd.merge(joined_linked, joined_linked_bkh, how = "left", on = ["dateeow","personid","portfolioid"])

        #To clear up some memory
        del joined_linked_bkh
        gc.collect()

        tempindex = pd.eval("joined_linked['boekhoudkoppeling'].isna()")
        joined_linked['has_account_overlay'] = 0
        joined_linked.loc[~tempindex, 'has_account_overlay'] = 1

        #TODO convert to other period: Use oen variable to aggregate and merge all values that cant be found
        # Or use this joined_linked['activitystatus'].mode()

        #TODO Concat files or keep apart


        print(f"****Finished linking timeseries sets at {utils.get_time()}****")


        """
        Used for checking data:
        """



 ##IMPORT AND CONVERT METHODS--------------------------------------------------------##

    def importSets(self, fileID, **readArgs):
        if fileID == "lpp" or fileID == "linkpersonportfolio.csv":
            return pd.read_csv(f"{self.indir}/linkpersonportfolio.csv", **readArgs)

        elif fileID == "bhk" or fileID == "portfolio_boekhoudkoppeling.csv":
            return pd.read_csv(f"{self.indir}/portfolio_boekhoudkoppeling.csv", **readArgs)

        elif fileID == "pin" or fileID == "portfolio_info.csv":
            return pd.read_csv(f"{self.indir}/portfolio_info.csv", **readArgs)

        elif fileID == "pst" or fileID == "portfolio_status.csv":
            return pd.read_csv(f"{self.indir}/portfolio_status.csv", **readArgs)

        elif fileID == "exp" or fileID == "experian.csv":
            return pd.read_csv(f"{self.indir}/experian.csv", **readArgs)

        elif fileID == 'cor' or fileID == 'corporate_details.csv':
            return pd.read_csv(f"{self.indir}/corporate_details.csv")

        ##Import Intermediate Files
        elif fileID == "expts" or fileID == "experianTS.csv":
            return utils.importChunk(f"{self.interdir}/experianTS.csv", 250000, **readArgs)

        elif fileID == "cored" or fileID == "df_corporate_details":
            self.df_corporate_details = pd.read_csv(f"{self.interdir}/corporate_details_processed.csv", **readArgs)

        elif fileID == "patot" or fileID == "experian.csv":
            self.df_pat = utils.importChunk(f"{self.interdir}/total_portfolio_activity.csv", 250000, **readArgs)

        elif fileID == "patsmp" or fileID == "total_portfolio_activity_sample.csv":
            self.df_pat = pd.read_csv(f"{self.interdir}/total_portfolio_activity_sample.csv", **readArgs)
        else:
            print("error importing")

    def exportEdited(self, fileID):
        errorMessage = ""
        if fileID == "expts":
            writeArgs = {"index": False}
            utils.exportChunk(self.df_expTS, 250000, f"{self.interdir}/experianTS.csv", **writeArgs)

        elif fileID == "patot" or fileID == "total_portfolio_activity.csv":
            if self.df_pat.empty:
                return print(errorMessage)
            writeArgs = {"index": False}
            utils.exportChunk(self.df_pat, 250000, f"{self.interdir}/total_portfolio_activity.csv", **writeArgs)

        elif fileID == "cored" or "df_corporate_details":
            if self.df_corporate_details.empty:
                return print(errorMessage)
            self.df_corporate_details.to_csv(f"{self.interdir}/corporate_details_processed.csv", index=False)

        elif fileID == "patsmp" or fileID == "total_portfolio_activity_sample.csv":
            if self.df_pat_sample.empty():
                return print(errorMessage)
            return print(errorMessage)
            self.df_pat_sample.to_csv(f"{self.interdir}/total_portfolio_activity_sample.csv", index=False)
        else:
            print("error exporting")

    def doConvertPAT(self):
        if self.df_pat.empty:
            return print("No df_pat to convert")
        self.df_pat = utils.doConvertFromDict(self.df_pat)