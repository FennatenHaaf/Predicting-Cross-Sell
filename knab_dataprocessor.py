"""
This code aims to create the data file that links portfolios to people

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import utils
import dataInsight
import declarationsFile
import gc
from tqdm import tqdm
from os import path
import re


class dataProcessor:

    def __init__(self,indir,interdir,outdir, 
                 quarterly = True,
                 start_date = "2018",
                 end_date = None,
                 save_intermediate=False,
                 print_info=False,
                 seed = 978391):
        """This method processes data provided by Knab"""
        
        #-------------------------INITIALISATION-----------------------
        self.indir= indir # location of input files
        self.interdir = interdir #location of intermediate files
        self.outdir = outdir # location of linked output files
        
        self.quarterly = quarterly # Quarterly periods if true, monthly otherwise
        self.start_date = start_date # when we start the time period
        self.end_date = end_date # when we end the time period
        
        self.save_intermediate = save_intermediate # location of linked output files
        self.print_info = print_info #Determines how verbose it is
        self.seed = seed
        
        #Declare data variables to check if data has been printed
        self.df_corporate_details = pd.DataFrame()
        self.df_pat = pd.DataFrame()
        self.df_pat_sample = pd.DataFrame()
        self.df_linked_ts_unc = pd.DataFrame()
        self.df_linked_ts_unc_sample = pd.DataFrame()
        self.df_linked_ts_time_converted = pd.DataFrame()
        
        # Declare other variables
        self.time_format = "%Y-%m-%d"
        self.endDate = datetime(2020, 12, 31) #Initalizes a last date for NaN values
        #TODO check if this is necessary, could also coerce errors or set it to
        # self.last_date later like I did in select_ids function?


      
    def link_data(self, outname = "base_linkinfo"):
        """Creates a dataset containing person IDs linked to their portfolio 
        ids and the corresponding portfolio information, as well as linking 
        them to corporate ids sharing the same portfolios and the corresponding 
        business details."""
        
        print("Linking portfolio information and business information to human IDs")
    
        #---------------------READING AND MERGING RAW DATA---------------------
                      
        if self.df_corporate_details.empty:
            self.processCorporateData() # create processed corporate dataset

        # Do a left join to add portfolio information  
        self.df_link = pd.read_csv(f"{self.indir}/linkpersonportfolio.csv").merge(
            pd.read_csv(f"{self.indir}/portfolio_info.csv"), 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 

        #Split the corporate persons off and rename to corporate id
        df_corporatelink = self.df_link[self.df_link["iscorporatepersonyn"]==1]
        df_corporatelink = df_corporatelink.loc[:,["personid","portfolioid"]]
        df_corporatelink = df_corporatelink.rename(columns={"personid": "corporateid",})
        
        # left join to find which human personids are linked to which corporate ids
        self.df_link = self.df_link.merge(df_corporatelink, 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 
        
        # Merge the file with business information on the corporate ids
        self.df_corporate_details = self.df_corporate_details.rename(columns={"personid": "corporateid"})
        self.df_link = self.df_link.merge(self.df_corporate_details, 
                                  how="left", left_on=["corporateid"],
                                  right_on=["corporateid"],) 
        if self.print_info:
            print("printing most common corporate ids:")
            dataInsight.mostCommon(self.df_link, "corporateid", 10)
    
        #------------------------ SAVE & RETURN  -------------------------
        if self.save_intermediate:
            utils.save_df_to_csv(self.df_link, self.interdir, 
                                  outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")



# =============================================================================
# Methods to select which IDs to use & make the base dataset
# =============================================================================


    def select_ids(self, subsample = True, sample_size = 500, 
                   finergy_segment=None,
                   outname = "base_experian",
                   filename = "valid_ids",
                   invalid ="invalid_ids",
                   use_file = True):
        """Selects a certain set of person IDs from the linking dataset, where:
            - information for ALL portfolios is present in the linking data
            - There is information for them in the Experian data for
            each time period (if time_series is TRUE) .
          We also take a subsample, if this is so specified."""

        print("Getting valid ids from the experian and the linking data")
        #TODO ook de optie geven om bijvoorbeeld mensen uit een specifieke 
        # sector te selecteren?
        
        # read in the experian data, which will be used as a base for everything
        self.df_experian = pd.read_csv(f"{self.indir}/experian.csv")
        # don't need this variable
        self.df_experian.drop(["business"], axis=1, inplace=True)
        
        #------------------------ GET DATES INSIGHT -------------------------
    
        # See what the most recent date is in the data 
        # Find the latest and oldest dates where a customer was changed or added 
        self.df_experian['valid_to_dateeow'] = pd.to_datetime(self.df_experian['valid_to_dateeow'])
        self.df_experian['valid_from_dateeow'] = pd.to_datetime(self.df_experian['valid_from_dateeow'])
        all_dates = pd.concat([self.df_experian['valid_to_dateeow'],
                               self.df_experian['valid_from_dateeow']])
        self.last_date = all_dates.max()
        self.first_date = all_dates.min()
                
        if self.print_info:
            print(f"Most recent data in Experian is from "\
                  f"{self.last_date.strftime(self.time_format)},")
            
            print(f"Oldest data in Experian is from "\
                  f"{self.first_date.strftime(self.time_format)}")
            
        # Also process end date and start date for the time period that was 
        # initialised
        if (self.end_date==None):
            self.end_date = self.last_date 
        else:
            self.end_date = datetime.strptime(self.end_date,self.time_format)
            assert self.end_date <= self.last_date, \
            "This is later than the most recent available data"
        
        self.start_date = datetime.strptime(self.start_date, self.time_format)

        #-------------------------- GET VALID IDS ---------------------------
       
        if (path.exists(f"{self.interdir}/{filename}.csv") & (use_file)):
            print("Reading from existing file of valid IDs")
            valid_ids = pd.read_csv(f"{self.interdir}/{filename}.csv").squeeze()
        else:
            print("creating new file of valid IDs:")
            valid_ids = self.get_valid_ids(filename,invalid)
            print(f"Finished and output saved, at {utils.get_time()}")
            
        print(f"got {len(valid_ids)} useable IDs from the Experian data")
        # Show how many there are of each finergy type
        dataInsight.plotFinergyCounts(self.df_experian,valid_ids)
        
        #------------------------ GET FINERGY SEGMENT ----------------------
        
        if finergy_segment != None:
            select = (self.df_experian["finergy_tp"] == finergy_segment)
            finergyids = self.df_experian["personid"][select]
            valid_ids = valid_ids[(valid_ids.isin(finergyids))]
            print(f"got {len(valid_ids)} IDs of finergy type {finergy_segment}")
        
        #-------------------- TAKE A SUBSAMPLE OF THE IDs ------------------
        
        if subsample:
            if(sample_size <= len(valid_ids)):
                print(f"Taking a subsample of {sample_size} IDs, at {utils.get_time()}.")
                valid_ids = valid_ids.sample(n = sample_size, 
                                             random_state = self.seed).reset_index(drop=True)
                print(f"Done at {utils.get_time()}.")
            else:
                print(f"The sample size ({sample_size}) can't be larger "\
                      f"than the number of IDs ({len(valid_ids)})")
                print("Continuing without taking subsample...")
               

        # Now we can select the base data
        self.base_df = self.df_experian[self.df_experian["personid"].isin(valid_ids)].copy()
        dataInsight.unique_IDs(self.base_df,"base Experian dataset") # show number of IDs
        
        #------------------ ADD A WEEK TO VALIDTO DATES  ------------------
        
        # To account for small gaps between validto and validfrom dates (which may be
        # due to administration), we add 7 days to the validto dates of experian
        select = ( ~(self.base_df["valid_to_dateeow"].isnull()) \
             &(self.base_df["valid_to_dateeow"] <= (self.last_date + timedelta(days=-7))) )
            
        self.base_df.loc[select,"valid_to_dateeow"] = \
            (self.base_df[["valid_to_dateeow"]]+timedelta(days=7))
                
        #------------------------ SAVE & RETURN -------------------------
        
        if self.save_intermediate:
            utils.save_df_to_csv(self.base_df, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")        
        
        print("-------------------------------------")
        
        
        
    def get_valid_ids(self, filename, invalid = "zzinvalid_ids",
                       use_file = True):
        """Gets the person IDs in the Experian dataset which have complete
        information for all of their portfolios and characteristics"""
        
        all_ids = pd.Series(self.df_experian["personid"].unique())
        len1 = len(all_ids)
        print(f"there are {len1} IDs in Experian dataset, dropping invalid ones")
        
        #------------------ ENSURE INFORMATION IN LINK DATA ------------------
        # we only want those ids for which the portfolio information is present, 
        # so dateinstroomweek should NOT be blank for ALL portfolios
        missing_info_ids= self.df_link["personid"][self.df_link["dateinstroomweek"].isnull()]        
        valid_ids = all_ids[~(all_ids.isin(missing_info_ids))] # remove ids from the set
        len2= len(valid_ids)
        print(f"{len1-len2} IDs dropped for not having information on ALL of their portfolios")

        # For the ones with corporate portfolio, we also want AT LEAST one of the
        # corporate IDs to have business information
        temp = self.df_link[["personid","businessType","corporateid"]].copy()
        temp = temp[~(temp["corporateid"].isnull())].copy()
        temp["info"] = [0 if pd.isnull(val) else 1 for val in temp["businessType"]]
        temp = temp[["personid", "info"]].groupby("personid").max().reset_index(level=0)
        # If the max is 0, then all of these values are missing, therefore we want to remove
        missing_info_ids2 = temp["personid"][temp["info"]==0]
        valid_ids = valid_ids[~(valid_ids.isin(missing_info_ids2))] # remove ids from the set
        len3= len(valid_ids)
        print(f"{len2-len3} IDs dropped for not having info on at least 1 of their businesses")
    
        #----------------- ENSURE INFORMATION IN EXPERIAN DATA ---------------
        # we want there to be experian data in every time period for the IDs 

        # Get a variable representing the last date until which an ID is valid
        maxending = self.df_experian[["valid_to_dateeow","personid"]].copy()
        maxending["valid_to_dateeow"] = maxending["valid_to_dateeow"].fillna(self.last_date).astype('datetime64[ns]')
        maxending = maxending.groupby("personid").max()
        maxending = maxending.rename(columns={"valid_to_dateeow": "valid_to_max",})

        # Get a variable representing the first date from which an ID is valid
        minstart = self.df_experian[["valid_from_dateeow","personid"]].copy()
        minstart = minstart.groupby("personid").min()
        minstart  = minstart.rename(columns={"valid_from_dateeow": "valid_from_min",})    
                             
        # add to experian data
        self.df_experian = self.df_experian.merge(maxending, how="left", 
                                left_on=["personid"], right_on=["personid"])
        self.df_experian = self.df_experian.merge(minstart, how="left", 
                                left_on=["personid"], right_on=["personid"])

        #-> The first validfrom date has to be BEFORE a certain starting date
        #-> The last validto date has to be AFTER a certain ending date    
        valid_ids_experian = self.df_experian["personid"][(self.df_experian["valid_to_max"]>= self.end_date) \
                                            & (self.df_experian["valid_from_min"]<= self.start_date)]        

        valid_ids_experian = valid_ids_experian.reset_index(level=0,drop=True)
        # Get the intersection with the previous set
        valid_ids = valid_ids[valid_ids.isin(valid_ids_experian)]
        len4= len(valid_ids)
        print(f"{len3-len4} IDs dropped for not being there for the full timeperiod")
        
        #-------------- MAKE SURE IDS ARE ACTIVE -------------
        
        # haal de active portfolio IDs op voor een specifieke periode
        active_portfolios = self.get_active_portfolios() 
        
        # We willen de person IDs waarvoor TEN MINSTE 1 portfolio ten minste
        # 1 activiteits entry heeft in de periode 2018-2020
        active_person_ids = \
        self.df_link["personid"][(self.df_link["portfolioid"].isin(active_portfolios))]  
        valid_ids = valid_ids[(valid_ids.isin(active_person_ids))]
        len5= len(valid_ids)
        print(f"{len4-len5} IDs dropped for not having at least one portfolio " \
              "in the transaction info in given time period")
        
        #------------NOW RUN THE TIME SERIES AND GET VALID IDS ---------------
        
        if (path.exists(f"{self.interdir}/{invalid}.csv") & (use_file)):
            invalid_ids = pd.read_csv(f"{self.interdir}/{invalid}.csv").squeeze()
        else:
            temp= self.df_experian[self.df_experian["personid"].isin(valid_ids)].copy()
    
            # To account for small gaps between validto and validfrom dates (which may be
            # due to administration), we add 7 days to the validto dates of experian
            select = ( ~(temp["valid_to_dateeow"].isnull()) \
                 &(temp["valid_to_dateeow"] <= (self.last_date + timedelta(days=-7))) )
            temp.loc[select,"valid_to_dateeow"] = \
                (temp[["valid_to_dateeow"]]+timedelta(days=7))
                
            date = self.get_last_day_period(self.start_date, next_period=False)
            end = self.get_last_day_period(self.end_date, next_period=False)
            
            invalid_ids = pd.Series()
            i=1
            starttime = utils.get_time()
            
            # Now, until we have gone through all of the periods make a cross-section
            while (date <= end):           
                print("=====================================================")                   
                print("Making data for each period to identify ids with missing data....")
                print(f"period {i}")
                i+=1
                
                #Make the cross-sectional dataset for this specific cross-date
                df_cross = self.get_time_slice(temp, date)
                
                # Make sure that IDs do not appear twice in df_cross 
                # -> We sort by date and keep the one with most recent valid_from
                df_cross = df_cross.sort_values("valid_from_dateeow")
                df_cross.drop_duplicates(subset=["personid"],
                              keep='last', inplace=True) 
                
                # make the cross section per portfolio
                df_cross_link = self.create_cross_section_perportfolio(df_cross, 
                                          date, outname = "portfoliolink_temp",
                                          save=False)
                
                # TODO this is not efficient, but: make crosssec per person 
                df_final = self.create_cross_section_perperson(df_cross,
                                                    df_cross_link, date,
                                                    outname = "df_final_temp",
                                                    save = False)
                
                # Add the IDs where data is missing to the IDs to be removed
                missing_experian= df_cross["personid"][df_cross["age_hh"].isnull()] 
                invalid_ids = invalid_ids.append(missing_experian.squeeze())
                missing_activity = df_cross_link["personid"][df_cross_link["saldobetalen"].isnull()]  
                invalid_ids = invalid_ids.append(missing_activity.squeeze())
                missing_portfolios = df_final["personid"][df_final["birthyear"].isnull()]  
                invalid_ids = invalid_ids.append(missing_portfolios.squeeze())
    
                invalid_ids = invalid_ids.drop_duplicates()
                
                # Now get the last day of the next period for our next cross-date
                date = self.get_last_day_period(date,next_period=True)
                
            print("=========== DONE GETTING MISSING IDS =============")
            endtime = utils.get_time()
            diff = utils.get_time_diff(starttime, endtime)
            print(f"Total time to make invalid IDs file: {diff}")
            print(f"Saving file to filename: {invalid}.csv")
            utils.save_df_to_csv(invalid_ids, self.interdir, 
                         f"{invalid}", add_time = False )
            print("=====================================================")    
        #--------------------------------------------------------------------    
        valid_ids = valid_ids[~(valid_ids.isin(invalid_ids))]
        len6= len(valid_ids)
        print(f"{len5-len6} IDs dropped for having either missing experian " \
              "or missing transaction data in one of the time periods")
        
        #------------------ SAVE & RETURN THE VALID IDS ---------------------
        utils.save_df_to_csv(valid_ids, self.interdir, 
                              f"{filename}", add_time = False )     
        return valid_ids
        
        
    
    def get_active_portfolios(self):
        """Gets all the portfolio IDS which appear in the transaction data within
        the specified time period AT LEAST once"""
        
        readArgs = {"usecols": ["portfolioid","dateeow"]}
        readlist = [f"{self.indir}/portfolio_activity_transaction_retail.csv",
                    f"{self.indir}/portfolio_activity_transaction_business.csv",
                    f"{self.indir}/portfolio_activity_business_2018.csv",
                    f"{self.indir}/portfolio_activity_retail_2018.csv",
                    f"{self.indir}/portfolio_activity_retail_2019.csv",
                    f"{self.indir}/portfolio_activity_retail_2019.csv",
                    f"{self.indir}/portfolio_activity_business_2020.csv",
                    f"{self.indir}/portfolio_activity_retail_2020.csv"]
        
        # We gebruiken start en end date
        ids = pd.DataFrame()
        for readpath in readlist:
           add_ids = pd.read_csv(readpath, **readArgs)
           ids = pd.concat([ids,add_ids])  
        ids = ids.drop_duplicates()
        
        # Get variables representing the first and last date the ID is in the
        # transaction data
        ids["dateeow"] = ids["dateeow"].astype('datetime64[ns]')
        maxdate = ids.groupby("portfolioid").max()
        maxdate = maxdate.rename(columns={"dateeow": "last_transac",})
        mindate = ids.groupby("portfolioid").min()
        mindate= mindate.rename(columns={"dateeow": "first_transac",})
        
        # add to IDs data
        idsunique = ids["portfolioid"].drop_duplicates().reset_index(level=0).copy() 
        idsunique = idsunique.merge(maxdate, how="left", left_on=["portfolioid"], 
                        right_on=["portfolioid"])    
        idsunique = idsunique.merge(mindate, how="left", left_on=["portfolioid"], 
                        right_on=["portfolioid"])
        
        #-> The first date has to be BEFORE a certain ending date
        #-> The last date has to be AFTER a certain starting date    
        valid_ids_transac = idsunique["portfolioid"][\
            (idsunique["first_transac"]<= self.end_date) \
            & (idsunique["last_transac"]>= self.start_date)]   
               
        valid_ids_transac = valid_ids_transac.reset_index(level=0,drop=True)
        return valid_ids_transac
        
    
  
        
  
  
# =============================================================================
# methods to make the final datasets ==========================================
# =============================================================================
    

    def time_series_from_cross(self, outname = "final_df"):
        """Run the cross-section dataset creation multiple times"""
    
        print("Starting date for the time series dataset:")
        date = self.get_last_day_period(self.start_date, next_period=False)
        print("Ending date for the time series dataset:")
        end = self.get_last_day_period(self.end_date, next_period=False)
        
        dflist = []
        
        # Now, until we have gone through all of the periods make a cross-section
        while (date <= end):        
            # get year and quarter for printing and naming files
            year = date.year
            quarter = ((date.month-1)//3)+1
            
            print("=====================================================")
            print(f"========== Getting cross-data for {year}Q{quarter} ============")
            print("=====================================================")
            
            #Make the cross-sectional dataset for this specific cross-date
            df_cross = self.get_time_slice(self.base_df, date)
            len1 = len(df_cross)
            # Make sure that IDs do not appear twice in df_cross 
            # -> We sort by date and keep the one with most recent valid_from
            df_cross = df_cross.sort_values("valid_from_dateeow")
            df_cross.drop_duplicates(subset=["personid"],
                          keep='last', inplace=True) 
            
            len2 = len(df_cross)
            print(f"{len1-len2} dropped because they were there double in this time period")
            
            df_cross_link = self.create_cross_section_perportfolio(df_cross, date, 
                                      outname = f"portfoliolink_{year}Q{quarter}")
            
            df_final = self.create_cross_section_perperson(df_cross, df_cross_link, date,
                                      outname = f"{outname}_{year}Q{quarter}")
            
            # Now get the last day of the next period for our next cross-date
            date = self.get_last_day_period(date,next_period=True)
            print(f"next date: {date}")
            
            dflist.append(df_final)
    
        print("=========== DONE MAKING TIME SERIES =============")
        return dflist
        
        
    def create_base_cross_section(self,
                                  date_string = None,
                                  next_period = False,
                                  outname = "cross_experian"):  
        """Method to create a cross-sectional dataset of the unique person ids.
        For this cross-sectional dataset we use only the information of the 
        customers who are a customer of Knab at the specified date. 
        We use their activity histories until the specified time and their
        characteristics at the specified time."""

        print(f"Creating cross-sectional data, at {utils.get_time()}")
        
        #-------------Check that date is entered correctly--------------
        
        if (date_string!=None): # If we take a date in the past for our cross-section
            date = datetime.strptime(date_string, self.time_format)
            assert date <= self.last_date, \
            "This is later than the most recent available data"
        else: # else we take the last day in the dataset as a standard
            date = self.last_date
            
        time_string = date.strftime(self.time_format)
        print(f"time-slice date is {time_string}")
        
        #-----------Taking a time-slice of Experian data for base dataframe-----------
        
        if (next_period):
            cross_date = self.get_last_day_period(date,next_period=True)
        else:
            cross_date = self.get_last_day_period(date,next_period=False)
        print(f"cross-date {cross_date.strftime(self.time_format)},")
    
        ## Now get the base dataset for this current period and the next 
        df_cross = self.get_time_slice(self.base_df,cross_date)
        print(f"Using data for {len(df_cross)} customers at this cross-section point")
        
        #------------------------ SAVE & RETURN -------------------------
        
        if self.save_intermediate:
            utils.save_df_to_csv(df_cross, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")
        
        return df_cross, cross_date
           
        print("-------------------------------------")



    def create_cross_section_perportfolio(self, df_cross, cross_date, 
                                          outname = "df_cross_portfoliolink",
                                          save = True): 
        """Creates a dataset of information for a set of people at a 
        specific time"""
        
        #----------- Portfolio + business information (per portfolio) --------
        print(f"Getting time-sliced linking data, at {utils.get_time()}")
        
        # Start by taking a cross-section of the dataset linking 
        # portfolio ids to person ids and to business+portfolio information
        df_cross_link = self.get_time_slice(self.df_link, cross_date,
                                            valid_to_string = 'validtodate',
                                            valid_from_string = 'validfromdate')
        
        # Take the same subsample of people as what we took from the experian data
        df_cross_link = df_cross_link.loc[df_cross_link["personid"].isin(
            df_cross["personid"].unique())]
        
    
        #----- Get only the ACTIVE portfolios (for which we have info) -----
        
        # get the unique portfolio ids which are still active at cross-section time
        df_portfolio_status = self.get_time_slice(pd.read_csv(f"{self.indir}/portfolio_status.csv"),
                                             cross_date,
                                             valid_to_string = 'outflow_date',
                                             valid_from_string = 'inflow_date')
        
        df_cross_link = df_cross_link.loc[df_cross_link["portfolioid"].isin(
            df_portfolio_status["portfolioid"].unique())]
        
        
        #-------------- Add Transaction information (per portfolio) --------------
        print(f"Summarizing transaction activity per portfolio, at {utils.get_time()}")
                
        temp_dataset = self.create_transaction_data_crosssection(cross_date)
        df_transactions = self.summarize_transactions(dataset = temp_dataset,
                                                      date = cross_date,
                                                      quarterly_period = self.quarterly)
        
        # Merge with the big linking dataset
        df_cross_link = df_cross_link.merge(df_transactions, 
                                  how="left", left_on=["portfolioid"],
                                  right_on=["portfolioid"],) 
        
        #-------------- Add activity information (per portfolio) -----------
        print(f"Summarizing activity per portfolio, at {utils.get_time()}")
                
        temp_dataset = self.create_activity_data_crosssection(cross_date)
        df_activity = self.summarize_activity(dataset = temp_dataset,
                                                    date = cross_date,
                                                    quarterly_period = self.quarterly)     
        
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
        print(f"Summarizing bookkeeping overlay data per portfolio, at {utils.get_time()}")
        
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
        df_portfolio_boekhoudkoppeling = df_portfolio_boekhoudkoppeling.groupby(["personid",
                                  "portfolioid"]).size().reset_index(name="accountoverlays")
        
        # merge with the portfolio link dataset
        df_cross_link = df_cross_link.merge(df_portfolio_boekhoudkoppeling,
                                            how="left", left_on=["personid","portfolioid"],
                                            right_on=["personid","portfolioid"])
        
        #------------------------ SAVE & RETURN -------------------------
        if save:
            if self.save_intermediate:
                utils.save_df_to_csv(df_cross_link, self.interdir, 
                                     outname, add_time = False )      
                print(f"Finished and output saved, at {utils.get_time()}")

        return df_cross_link  
        print("-------------------------------------")
        
        
        
        
        
    def create_cross_section_perperson(self, df_cross, df_cross_link,
                                       cross_date, outname = "final_df",
                                       save = True):  
        """This function takes a dataset linking each person ID to
        information for each portfolio separately, and returns a dataset
        where all the portfolio information is aggregated per person ID"""
       
        print(f"Summarizing all information per ID, at {utils.get_time()}")
        
        #-------------------------------------------------------------------
        
        # Create a dictionary of the data separated by portfolio type
        df_cross_link_dict = {}
        df_cross_link_dict['business'] = df_cross_link[df_cross_link["type"]=="Corporate Portfolio"]
        df_cross_link_dict['retail'] = df_cross_link[(df_cross_link["type"]=="Private Portfolio")\
                                             & (df_cross_link["enofyn"]==0)]
        df_cross_link_dict['joint'] = df_cross_link[(df_cross_link["type"]=="Private Portfolio")\
                                             & (df_cross_link["enofyn"]==1)]
        
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
            
            # We pakken het MAXIMUM om de meest actieve rekening weer te geven
            indicator = indicator.groupby("personid").max()
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            # Maar voor de saldos en aantal rekeningen pakken we de SOM over alles 
            indicator = df.loc[:,["personid"]]
            indicator[f"saldototaal_{name}"] = df.loc[:,"saldototaal"]
            indicator[f"betalenyn_{name}"] = df.loc[:,"betalenyn"]
            indicator[f"depositoyn_{name}"] = df.loc[:,"depositoyn"]
            indicator[f"flexibelsparenyn_{name}"] = df.loc[:,"flexibelsparenyn"]
            indicator[f"kwartaalsparenyn_{name}"] = df.loc[:,"kwartaalsparenyn"]
            
            indicator = indicator.groupby("personid").sum()
            df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
            
            
            #------ Variables that are specific to portfoliotype -------
            if name == 'joint': 
                # Add gender for the joint portfolio
                indicator = df.loc[:,["personid"]]
                indicator[f"geslacht_{name}"] = df.loc[:,"geslacht"]
                indicator = indicator.drop_duplicates(subset=["personid"])
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
                
                # TODO: Possibly we want to do other things for the joint porfolio 
                # data to take into account that joint accounts may be more active?
            
            if name == 'business':
                # See if they have a bookkeeping overlay
                indicator = df.loc[:,["personid"]][df["accountoverlays"]>0] 
                indicator["accountoverlay"]=1 # make all of the columns one and give it the name
                # Sum to find for how many business portfolios there is an 
                # account overlay
                indicator = indicator.groupby("personid").sum() 
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
                            
                # ------ Now INCORPORATE the business characteristics --------:
                    
                # We take the MEAN age of the business in years
                #(could also take the MAX, to get the oldest business)
                indicator = df.loc[:,["personid", "businessAgeInYears"]]
                indicator = indicator.groupby("personid").mean() 
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)
                
                # We process the SBI codes and types by taking the type itself
                # if the person is linked to one type of business, but we take
                # type "meerdere" otherwise
                
                temp = df.loc[:,["personid", "SBIcode", "SBIname"]]
                indicator = self.aggregateBusinessPerPerson(temp, 
                                        count_name = "aantal_SBI")
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)    
                
                #Doe precies hetzelfde met de sector 
                temp = df.loc[:,["personid","SBIsector","SBIsectorName"]]
                indicator = self.aggregateBusinessPerPerson(temp, 
                                        count_name = "aantal_sector")
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)    
                
                #Doe hetzelfde met de Business Type 
                temp = df.loc[:,["personid", "businessType"]]
                indicator =  self.aggregateBusinessPerPerson(temp, 
                                        count_name = "aantal_types")
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)    
            
        #----------- Merge remaining variables --------
        
        # Get the personid and birthyear which were stored in crosslink 
        # - first drop the duplicates
        characteristics = df_cross_link.loc[:,["personid","birthyear","geslacht",
                "enofyn"]]
        
        # Sort the characteristics by enofyn, so that the joint portfolios appear later.
        # then drop duplicates and keep the first entries that appear.
        characteristics = characteristics.sort_values("enofyn")
        characteristics = characteristics.drop_duplicates(subset=["personid"],
                          keep='first', inplace=False) 
        
        characteristics.loc[(characteristics["geslacht"]=="Mannen"), "geslacht"]="Man"
        
        # Now merge back with df_cross                              
        df_cross= df_cross.merge(characteristics[["personid","birthyear","geslacht"]], 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],) 
        
        # TODO: eventueel nog duration variables toevoegen, zoals de tijd sinds
        # de meest recente transaction, tijd sinds klant worden
                
        #------------------------ SAVE & RETURN -------------------------
        
        if save:
            utils.save_df_to_csv(df_cross, self.interdir, 
                                 outname, add_time = False )      
            print(f"Finished and output saved, at {utils.get_time()}")
            print("===============================================")
        return df_cross
        
        

    def aggregateBusinessPerPerson(self, temp, count_name):
        """ Aggregates some business characteristics by changing it to type
        'multiple' if the person has business portfolios from multiple sectors"""
        
        temp = temp.drop_duplicates()
        # Pak nu de IDs en per ID hoe vaak hij voor komt in temp
        indicator = temp["personid"].value_counts().rename_axis(\
                    'personid').to_frame(count_name).reset_index(level=0)
        
        # Merge de data die maar 1 keer voorkomt en dus maar 1 type sector heeft
        IDtemp = indicator["personid"][indicator[count_name]==1]
        temp = temp[temp["personid"].isin(IDtemp)]
        temp= temp.fillna("missing")
        indicator= indicator.merge(temp, 
                          how="left", left_on=["personid"],
                          right_on=["personid"],)
          
        # Vervolgens, voor mensen die nog wel meerdere keren voorkomen
        # willen we type 'meerdere' geven 
        indicator = indicator.fillna("meerdere")
        indicator = indicator.replace("missing", np.nan)
        # TODO wat als voor de meerdere codes de data eigenlijk ook missing is?
        
        return indicator




# =============================================================================
# #some helper methods that are used to handle time ===========================
# =============================================================================


    def get_last_day_period(self, date, next_period = False):
        """Returns the last day of the period that contains our date, by getting 
        the day BEFORE the FIRST day of the next period """
        
        if self.quarterly:
            quarter = ((date.month-1)//3)+1  #the quarter of our time slice 
   
            if (next_period):
                print(f"quarter is {date.year}Q{quarter}, we take the last date of the NEXT quarter")
                if (quarter<3):
                    end_date = datetime(date.year, (3*(quarter+1))%12 +1, 1) + timedelta(days=-1)
                else: # dates of the next period cross onto the next year
                    end_date = datetime(date.year+1, (3*(quarter+1))%12 +1, 1) + timedelta(days=-1)
            else:
                print(f"quarter is {date.year}Q{quarter}, we take the last date of the quarter")
                if(quarter<4):
                    end_date = datetime(date.year, (3*quarter)%12 +1, 1) + timedelta(days=-1)
                else: #dates of the next period cross onto the next year
                    end_date = datetime(date.year+1, (3*quarter)%12 +1, 1) + timedelta(days=-1)
            
        else: # We work in periods of months
            month = date.month
            if (next_period):
                print(f"month is {date.year}M{date.month}, we take the last date of the NEXT month")
                if (month<11):# dates of the next period cross onto the next year
                    end_date = datetime(date.year, (month+1)%12 +1, 1) + timedelta(days=-1)
                else:
                    end_date = datetime(date.year+1, (month+1)%12 +1, 1) + timedelta(days=-1)
            else:
                print(f"month is {date.year}M{date.month}, we take the last date of the month")
                if (month<12):# dates of the next period cross onto the next year
                    end_date= datetime(date.year, (month)%12 +1, 1) + timedelta(days=-1)
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
    


    


    
# =============================================================================
# methods to process corporate data ===========================================
# =============================================================================

    def processCorporateData(self):
        """processes the corporate information data and creates some extra
        variables"""

        # -------------READ IN CORPORATE DETAILS AND PROCESS------------
        self.df_corporate_details = pd.read_csv(f"{self.indir}/corporate_details.csv")

        if self.print_info:  # Print NaNs and most common values
            nameList = ["personid", "subtype", "name"]
            nameList2 = ["personid", "birthday", "subtype", "code", "name"]
            print("unique number of businessID's in corporate data :", self.df_corporate_details["subtype"].unique().shape)
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

        if self.print_info:  # show dimensions
            print("shape of current data", self.df_corporate_details)

        # ---------- CREATE foundingDate, companyAgeInDays AND foundingYear--------

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

        # -------------PROCESS SBI CODES------------
        # TODO: add comments to describe what is happening here

        SBI_2019Data = pd.read_excel("SBI_2019.xlsx")

        tempString = "SBIcode"
        tempString2 = "SBIname"

        SBI_2019DataEdited = SBI_2019Data.copy()
        SBI_2019DataEdited.rename(columns={
            "Unnamed: 0": tempString,
            "Standaard Bedrijfsindeling 2008 - update 2019 ":
                tempString2}, inplace=True)

        SBI_2019DataEdited = SBI_2019DataEdited[[tempString, tempString2]]
        ztestc21 = "SBI_2019New = SBI_2019DataEdited.copy()"
        ztestc22 = "SBI_2019DataEdited = SBI_2019New.copy()"
        ztestc23 = "examp2 = pd.DataFrame([SBI_2019DataEdited.loc[78,tempString]])"

        SBI_2019DataEdited.dropna(subset=[tempString], inplace=True)
        SBI_2019DataEdited.reset_index(drop=True, inplace=True)

        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].astype('string') + "a"
        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].str.replace(u"\xa0", u".0", regex=True)
        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].str.replace("a", "", regex=True)
        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].str.replace("\.0", "", regex=True)
        SBI_2019DataEdited.dropna(subset=[tempString], inplace=True)
        SBI_2019DataEdited = SBI_2019DataEdited.loc[:, [tempString, tempString2]]
        codesList = list(SBI_2019DataEdited[tempString].unique())

        sectorList = []
        tempList = ["A", 0, 0]
        tempList2 = ["A"]
        for value in codesList[1:]:
            try:
                text = str(value)
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

            try:
                if value[0] == '0':
                    edited_value = value[1]
                else:
                    edited_value = value
                tempList[2] = int(edited_value[:2])
            except:
                pass

        sectorList.append(tempList.copy())
        sectorData = pd.DataFrame(tempList2)
        sectorData.rename({0: tempString}, axis=1, inplace=True)
        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].str.replace(" ", "")
        sectorData = pd.merge(sectorData, SBI_2019DataEdited[[tempString, tempString2]], how="inner", on=tempString)
        sectorData.rename({tempString: "SBIsector", tempString2: "SBIsectorName"}, axis=1, inplace=True)
        SBI_2019DataEdited.dropna(inplace=True)
        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].str.replace(" ", "")
        SBI_2019DataEdited.reset_index(drop=True, inplace=True)

        def cleanSBI(x):
            try:
                if x[0] == '0':
                    x = x[1:]
                else:
                    pass
                return int(x)
            except:
                return np.nan

        SBI_2019DataEdited[tempString] = SBI_2019DataEdited[tempString].apply(lambda x: cleanSBI(x))
        SBI_2019DataEdited = SBI_2019DataEdited[SBI_2019DataEdited[tempString] <= 99]
        SBI_2019DataEdited.drop_duplicates(inplace=True)
        SBI_2019DataEdited[tempString2] = SBI_2019DataEdited[tempString2].astype("str")

        SBI_2019DataEdited["SBIsector"] = np.nan
        # chr, ord, < <=
        for values in sectorList:
            tempIndex = (values[1] < SBI_2019DataEdited[tempString]) & (SBI_2019DataEdited[tempString] <= values[2])
            SBI_2019DataEdited.loc[tempIndex, "SBIsector"] = values[0]

        SBI_2019DataEdited = pd.merge(SBI_2019DataEdited, sectorData, how="left", on="SBIsector")
        SBI_2019DataEdited.drop_duplicates(inplace=True)
        SBI_2019DataEdited.reset_index(inplace=True, drop=True)
        SBI_2019DataEdited = SBI_2019DataEdited.append(
            {tempString: 0, tempString2: 'Onbekend', 'SBIsector': 'Z', 'SBIsectorName': 'Onbekend'},
            ignore_index=True)

        tempString = "SBIcode"
        self.df_corporate_details[tempString] = self.df_corporate_details["businessSector"].str[:2]
        self.df_corporate_details[tempString] = pd.to_numeric(self.df_corporate_details[tempString],
                                                              downcast="unsigned")

        self.df_corporate_details = pd.merge(self.df_corporate_details, SBI_2019DataEdited, how="inner",
                                             on="SBIcode")
        self.df_corporate_details.drop(["code", "businessSector", "birthday"], axis=1, inplace=True)



# =============================================================================
# methods to create transaction & activity data ===============================
# =============================================================================
    
    def create_transaction_data_crosssection(self, date = None):
        """creates transaction data based on the cross-section for a specific date"""
        
        # Bepaalde variabelen zijn in sommige jaren altijd 0 dus gebruiken we niet
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







# =============================================================================
# linkTimeSets=================================================================
# =============================================================================

    def link_data_to_timeseries(self, use_sample = False, select_col = True):
        """
        Links the already imported Portfolio Activity File to other data files.
        This method will produce a dataset with activity of portfolios linked to personid's
        associated with this portfolio.

        This happens in these steps:
        -> Combine Linkperson Portfolio to experian on personid
        -> Combine This combined dataset to portfolio activity

        -> Combine corporate details and linkpersonportfolio
        -> Join this combined dataset to portfolio activity

        -> Join boekhoudkoppelingen to this larger dataset

        -> An indicator is created to follow which
        row contains either corporate info, experian info or
        account overlay info.


        It is possible to sample the a
        """

        print(f"****linking timeseries sets, starting at {utils.get_time()}****")


        if self.df_corporate_details.empty:
            try:
                self.importSets('cored')
            except:
                self.processCorporateData()

        if self.df_pat.empty:
            if use_sample:
                try:
                    self.importSets('patsmp', select_col= select_col)
                except:
                    self.importPortfolioActivity(convertData= True,selectColumns= select_col)
                    self.portfolioActivitySampler()
                    self.df_pat = self.df_pat_sample
            else:
                try:
                    self.importSets('patot')
                except:
                    self.importPortfolioActivity(convertData=True, selectColumns=True)

        df_lpp = pd.read_csv(f'{self.indir}/linkpersonportfolio.csv')
        df_exp = pd.read_csv(f'{self.indir}/experian.csv')
        df_bhk = pd.read_csv(f'{self.indir}/portfolio_boekhoudkoppeling.csv')
        df_pin = pd.read_csv(f'{self.indir}/portfolio_info.csv')
        df_cor = self.df_corporate_details


        '''Convert total portfolio activty and create a list to check if values can be found in this list. This unique value list
        #Is also used to ensure that LPP is sampled to reduce computation and get a grip on if the right data is parsed. After 
        That the df_pat file is converted to less memory intensive datatypes and an enddate is given to ensure a consistent end of 2020 
        ending of portofolios.
        '''
        pat_unique = self.df_pat['portfolioid'].unique()
        self.df_pat = utils.doConvertFromDict(self.df_pat)
        # self.df_pat = utils.select_time_in_data(self.df_pat, 'dateeow',period_to_use= 'Q', start = '2018Q1')

        self.df_pat.loc[self.df_pat["dateeow"] == "2021-01-03 00:00:00", "dateeow"] = self.endDate


        if use_sample == True:
            # SAMPLEFORTEST
            df_lpp = df_lpp.loc[pd.eval("df_lpp['portfolioid'].isin(pat_unique)"), :]
            # SAMPLEFORTESTEND

        '''
        Creating an index and indicator to see which personid'' are linked to a portfolio with business and retail ID's
        '''
        lpp_columns_to_use = ['personid', 'portfolioid', "iscorporatepersonyn", 'validfromdate']
        df_lpp = df_lpp[lpp_columns_to_use]
        df_lpp.rename({'validfromdate': 'validfromdate_lpp'}, axis=1, inplace = True)
        df_lpp = utils.doConvertFromDict(df_lpp)

        df_lpp_table_port_corp_retail = df_lpp.groupby("portfolioid")["iscorporatepersonyn"].mean()
        df_lpp_table_port_corp_retail = df_lpp_table_port_corp_retail.reset_index()
        df_lpp_table_port_corp_retail.loc[:, "indicator_corp_and_retail"] = 0
        lpp_index_both = pd.eval(" (df_lpp_table_port_corp_retail['iscorporatepersonyn'] > 0) & "
                                 "(df_lpp_table_port_corp_retail['iscorporatepersonyn'] < 1) ")
        df_lpp_table_port_corp_retail.loc[lpp_index_both, "indicator_corp_and_retail"] = 1
        df_lpp_table_port_corp_retail.drop("iscorporatepersonyn", axis=1, inplace=True)
        df_lpp = pd.merge(df_lpp, df_lpp_table_port_corp_retail, on="portfolioid")

        #Different indices to be used for checks and conversion
        index_link_cr = pd.eval("df_lpp['indicator_corp_and_retail'] == 1")
        index_corp_pers = pd.eval("df_lpp['iscorporatepersonyn'] == 1")
        portid_link_cr = df_lpp.loc[index_link_cr, "portfolioid"].unique()
        persid_link_cr = df_lpp.loc[index_link_cr, "personid"].unique()
        persid_no_link_cr = df_lpp.loc[~index_link_cr, "personid"].unique()
        persid_no_link_cr = persid_no_link_cr[~persid_no_link_cr.isin(persid_link_cr)]
        portid_no_link_cr = df_lpp.loc[~index_link_cr, "portfolioid"].unique()
        port_no_link_cr = portid_no_link_cr[~portid_no_link_cr.isin(portid_link_cr)]


        # TODO VEEL PORTFOLIOS DIE ALS EEN PERSOON STAAN HEBBEN WEL ALLEEN MAN OF VROUW ER BIJ
        # Methods used to look into the data
        ztest3a1 = "dataInsight.mostCommon(df_pin['enofyn'] == 0], 'geslacht',5)"
        ztest3a2 = "dataInsight.mostCommon(df_pin['enofyn'] == 1], 'geslacht',5)"
        ztest3a3 = """only_man_female_pin = df_pin[pd.eval("df_pin['geslacht'] != 'Man' & df_pin['geslacht'] != 'Vrouw'")]"""
        ztest3a4 = "(df_pin['portfolioid'].value_counts()>1).sum()"
        ztest3a5 = "pat_pin_dates_merge = self.df_pat.groupby('portfolioid')['dateeow'].min()"
        ztest3a6 = "pat_pin_dates_merge = pd.merge(pat_pin_dates_merge, df_pin[['portfolioid','dateinstroomweek']], on = 'portfolioid')"


        """"
        Link with Portfolio Info Data
        """
        #SAMPLING
        if use_sample == True:
            df_pin = df_pin[pd.eval("df_pin['portfolioid'].isin(pat_unique)")]

        print(self.df_pat.shape)
        joined_linked = self.df_pat.copy()
        df_pin = utils.doConvertFromDict(df_pin)

        selected_columns_pin = ['portfolioid', 'dateinstroomweek', 'birthyear', 'geslacht', 'type', 'enofyn']
        joined_linked = pd.merge(joined_linked, df_pin[selected_columns_pin],
                                 on='portfolioid')  # Are a few NA rows which start on the last date,
        # discarded for now

        portfolio_start_date_list = joined_linked.groupby('portfolioid').aggregate({'dateeow': np.min, 'dateinstroomweek': np.min})
        portfolio_start_date_list.reset_index(inplace=True)
        portfolio_start_date_list.dropna(inplace=True)
        correct_date_portfolio_list = portfolio_start_date_list.query("dateeow <= dateinstroomweek")['portfolioid']
        joined_linked['has_correct_startdate'] = 0
        joined_linked.loc[pd.eval("joined_linked['portfolioid'].isin(correct_date_portfolio_list)"),'has_correct_startdate'] = 1

        ''''
        EXPERIAN, LPP : Merge Experian and LPP. After that merge these sets to the larger file.  
        '''

        # SAMPLEFORTEST START
        if use_sample == True:
            df_exp = df_exp[(df_exp['personid'].isin(persid_link_cr)) | (df_exp['personid'].isin(persid_no_link_cr))].copy()
        # END SAMPLE

        # CHECK IF VALUES CAN BOTH HAVE INFO
        df_exp['valid_to_dateeow'] = pd.to_datetime(df_exp['valid_to_dateeow'])
        df_exp['valid_to_dateeow'].fillna(self.endDate, inplace=True)
        df_exp.sort_values(['valid_to_dateeow', 'personid'], inplace=True)
        df_exp.dropna(axis=0, subset=df_exp.columns[3:].tolist(), how='all', inplace=True)
        index_only_buss_finergy = df_exp['age_hh'].isna()
        # exp_only_buss_finergy = df_exp[index_only_buss_finergy].copy()
        df_exp = df_exp[~index_only_buss_finergy]
        df_exp = utils.doConvertFromDict(df_exp)

        df_exp['retail_id_with_corp_and_retail'] = 0
        df_exp.loc[df_exp['personid'].isin(persid_link_cr), 'retail_id_with_corp_and_retail'] = 1


        # Link self.df_pat and
        exp_lpp_joined = pd.merge(df_exp, df_lpp, on=['personid'])
        exp_lpp_joined.sort_values(['valid_to_dateeow', 'personid', 'portfolioid'], inplace=True)
        self.df_pat.sort_values(['dateeow', 'portfolioid'], inplace=True)

        # Merge joined by picking records to match that are before valid_to_dateeow, not matching records or expired records discarded
        self.df_linked_ts_unc = pd.merge(self.df_pat, exp_lpp_joined, on="portfolioid")
        temp_index = pd.eval("(self.df_linked_ts_unc['valid_from_dateeow'] <= self.df_linked_ts_unc['dateeow']) & \
                               (self.df_linked_ts_unc['valid_to_dateeow'] >= self.df_linked_ts_unc['dateeow'])")
        self.df_linked_ts_unc = self.df_linked_ts_unc[temp_index]

        temp_index = pd.eval("self.df_pat['portfolioid'].isin(self.df_linked_ts_unc['portfolioid'])")
        self.df_linked_ts_unc = pd.concat([self.df_linked_ts_unc, self.df_pat[~temp_index].copy()], ignore_index=True)

        del exp_lpp_joined #delete to clear more memory
        gc.collect()
        ''''
        CORPORATE DETAILS AND LINK PERSON PORTFOLIO. Merge Corporate details and Link Person Portfolio. After that, 
        merge this to the larger set. 
        '''
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

        cor_lpp_linked['business_id_with_corp_and_retail'] = 0
        cor_lpp_linked.loc[cor_lpp_linked['personid'].isin(persid_link_cr), 'business_id_with_corp_and_retail'] = 1

        # Merge corp_lpp with large joined table

        cor_lpp_linked.rename({'personid': 'businessid'}, axis=1, inplace = True)
        print(f"before merge dimension of self.df_linked_ts_unc : {self.df_linked_ts_unc.shape} and dimension of cor_lpp : {cor_lpp_linked.shape}")
        self.df_linked_ts_unc = pd.merge(self.df_linked_ts_unc, cor_lpp_linked, how="left", on="portfolioid", suffixes=['', '_business'])
        print(f"after merge dimension: {self.df_linked_ts_unc.shape}")

        del cor_lpp_linked #save memory
        gc.collect()
        """
        Merge Boekhoudkoppeling to large file
        """
        ##SAMPLESTART
        df_bhk[pd.eval("df_bhk['portfolioid'].isin(pat_unique)")].shape
        ##SAMPLEEND

        df_bhk = utils.doConvertFromDict(df_bhk)
        df_bhk.loc[df_bhk['valid_to_dateeow'].isna(), 'valid_to_dateeow'] = self.endDate
        df_bhk.drop(['accountid', 'accountoverlayid'], axis=1, inplace=True)  # Drop unused vars
        df_bhk.drop_duplicates(inplace=True)
        df_bhk.sort_values(['valid_to_dateeow', 'personid', 'portfolioid'], inplace=True)
        self.df_linked_ts_unc.sort_values(['dateeow', 'personid', 'portfolioid'])

        # Chosen to merge on person rather than portfolio
        person_id_in_bhk = df_bhk['personid'].unique()
        before_merge_index_bhk = self.df_linked_ts_unc['personid'].isin(person_id_in_bhk)

        print("with bhk dimensions before merge :", self.df_linked_ts_unc.shape)

        templist = ['dateeow', 'personid', 'portfolioid']
        self.df_linked_ts_unc_bkh = pd.merge(self.df_linked_ts_unc.loc[before_merge_index_bhk, templist], df_bhk, on=['personid', 'portfolioid'])

        self.df_linked_ts_unc_bkh.query("valid_from_dateeow <= dateeow <= valid_to_dateeow", inplace = True)
        self.df_linked_ts_unc_bkh.query("valid_from_dateeow != valid_to_dateeow", inplace = True) # Probably erronous that bhk can be active one day only

        print('Size file after merge :', self.df_linked_ts_unc_bkh.shape)
        print('Not NA after merge :', self.df_linked_ts_unc_bkh["boekhoudkoppeling"].notna().sum())
        print('Not NA after merge :', self.df_linked_ts_unc_bkh["valid_from_dateeow"].notna().sum())

        self.df_linked_ts_unc_bkh.drop(['valid_from_dateeow', 'valid_to_dateeow'], axis=1, inplace=True)
        self.df_linked_ts_unc = pd.merge(self.df_linked_ts_unc, self.df_linked_ts_unc_bkh, how="left", on=["dateeow", "personid", "portfolioid"])
        print(f"Final Joined File after merge {self.df_linked_ts_unc.shape}")

        del self.df_linked_ts_unc_bkh #clear up some memory
        gc.collect()

        #Final indices to add
        self.df_linked_ts_unc['has_account_overlay'] = 0
        self.df_linked_ts_unc.loc[pd.eval("self.df_linked_ts_unc['boekhoudkoppeling'].notna()"), 'has_account_overlay'] = 1

        self.df_linked_ts_unc['has_business_id'] = 0
        self.df_linked_ts_unc.loc[pd.eval("self.df_linked_ts_unc['businessid'].notna()"), 'has_business_id'] = 1

        self.df_linked_ts_unc['has_experian_data'] = 0
        self.df_linked_ts_unc.loc[pd.eval("self.df_linked_ts_unc['finergy_tp'].notna()"), 'has_experian_data'] = 1

        no_extra_information_index = self.df_linked_ts_unc.eval('has_business_id == 0 & has_account_overlay == 0 & has_experian_data == 0')

        self.df_linked_ts_unc = self.df_linked_ts_unc[~no_extra_information_index].copy()
        print(f"the dimension of the linked file is {self.df_linked_ts_unc.shape} and the dimension of observations with no experian, "
              f"corporate or accountoverlay data is {no_extra_information_index.shape} ")
        print(f"****Finished linking timeseries sets at {utils.get_time()}****")

    ###################----------------------------------------------------------------------------

    def convert_time_linked_time_series(self, period_to_convert_to ="Q", period_to_use ="All", use_sample = False, select_col = True):
        ''''
        Aggregation of Data based on time and personid
        '''
        print(f"****Importing needed files at {utils.get_time()}****")
        if self.df_linked_ts_unc.empty: #check if linked time series has been defined
            if use_sample:
                try:
                    self.importSets('ltsuncsmp',select_col= select_col)
                except:
                    self.link_data_to_timeseries()
                    self.linked_ts_unconverted_sampler()
            else:
                try:
                    self.importSets('ltsunc', select_col= select_col)
                except:
                    self.link_data_to_timeseries()

        print(f"****Started converting period of timeset {utils.get_time()}****")
        self.df_linked_ts_time_converted = self.df_linked_ts_unc.copy()
        self.df_linked_ts_time_converted = pd.get_dummies(self.df_linked_ts_time_converted, columns=['activitystatus'], prefix="indicator")
        time_convert_dict = utils.doDictIntersect(self.df_linked_ts_time_converted.columns, declarationsFile.getTimeConvertDict())

        self.df_linked_ts_time_converted = utils.doConvertFromDict(self.df_linked_ts_time_converted, ignore_errors=True)
        self.df_linked_ts_time_converted['converted_period'] = self.df_linked_ts_time_converted['dateeow'].dt.to_period(
            period_to_convert_to)


        self.df_linked_ts_time_converted = self.df_linked_ts_time_converted. \
            groupby(['personid', 'portfolioid', 'converted_period'], observed=True, as_index = False).aggregate(time_convert_dict)
        print(f"****Finished converting time period of time series at  {utils.get_time()}****")

    def aggregate_over_personid_time_series(self):
        if self.df_linked_ts_time_converted.empty:
            self.convert_time_linked_time_series()

        print(f"****Started aggregating data of timeseries at {utils.get_time()}****")

        id_aggregate_dict = utils.doDictIntersect(self.df_linked_ts_time_converted.columns, declarationsFile.getPersonAggregateDict())

        self.df_lts_agg = self.df_linked_ts_time_converted.groupby(['personid', 'converted_period'], observed=True,
                                                                   as_index = False).aggregate(
            id_aggregate_dict)

        new_name_list = []
        prev_name = ""
        #Change several things in columns names
        for value in self.df_lts_agg.columns:
            new_name = value[0]
            if prev_name == value[0]:
                new_name += "_" + value[1]
            prev_name = value[0]
            new_name = new_name.replace('<lambda_0>', 'mode')
            new_name = new_name.rstrip('_')
            new_name_list.append(new_name)
        self.df_lts_agg.columns = new_name_list


        print(f"****Finished aggregating data of timeseries at {utils.get_time()}****")
        pass

    def importPortfolioActivity(self, convertData=False, selectColumns=False,
                                discardPat=False, **readArgs):

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
        pa1420 = utils.importAndConcat(tempList, chunkSize=250000, **readArgsAct)
        print(pa1420.shape, " are the dimensions of pa before merge 14-20")

        tempList = [f"{self.indir}/portfolio_activity_transaction_business.csv",
                    f"{self.indir}/portfolio_activity_transaction_retail.csv"]
        pat1420 = utils.importAndConcat(tempList, chunkSize=250000, **readArgsTrans)
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


        ##IMPORT AND CONVERT METHODS--------------------------------------------------------##

    def importSets(self, fileID, select_col = False, addition_to_name = "", **readArgs):
        def remove_datetime_from_dict(data):
            a_dict = utils.doDictIntersect(data.columns, declarationsFile.getConvertDict())
            date_list = []
            return_dict = {}
            for value in a_dict:
                if re.match(r"datetime.", a_dict[value]):
                    date_list.append(value)
                else:
                    if re.match(r'.?int[8,16]', a_dict[value]):
                        return_dict[value] = 'float16'
                    elif re.match(r'.?int[32]', a_dict[value]):
                        return_dict[value] = 'float32'
                    else:
                        return_dict[value] = a_dict[value]

            return return_dict, date_list


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
        elif fileID == "ltsunc" or fileID == "linked_ts_unconverted.csv":

            # readArgs = {**readArgs, 'low_memory': False}
            self.df_linked_ts_unc = pd.read_csv(f"{self.interdir}/linked_ts_unconverted{addition_to_name}.csv", nrows=0, **readArgs)
            convert_at_import_dict, time_parse_list = remove_datetime_from_dict(self.df_linked_ts_unc)
            if select_col:
                col_to_parse = declarationsFile.getPatColToParseTSunc()
                readArgs = {**readArgs, 'usecols' : col_to_parse}
                convert_at_import_dict= utils.doDictIntersect(col_to_parse,convert_at_import_dict)
                time_parse_list = utils.doListIntersect(time_parse_list,col_to_parse)

            readArgs = {**readArgs, 'dtype': convert_at_import_dict}
            if len(time_parse_list) > 0:
                readArgs = {**readArgs,'parse_dates':time_parse_list}
            try:
                self.df_linked_ts_unc = utils.importChunk(f"{self.interdir}/linked_ts_unconverted{addition_to_name}.csv", 250000, **readArgs)
            except Exception as e:
                print(type(e))

        elif fileID == "ltsuncsmp" or fileID == "linked_ts_unconverted_sample.csv":
            if select_col:
                readArgs = {**readArgs, 'usecols' : declarationsFile.getColToParseLTSunc()}
            readArgs = {**readArgs, 'low_memory': False}
            self.df_linked_ts_unc = utils.importChunk(f"{self.interdir}/linked_ts_unconverted_sample{addition_to_name}.csv", 250000, **readArgs)

        elif fileID == "cored" or fileID == "df_corporate_details":
            self.df_corporate_details = pd.read_csv(f"{self.interdir}/corporate_details_processed.csv", **readArgs)

        elif fileID == "patot" or fileID == "total_portfolio_activity.csv":
            if select_col:
                readArgs = {**readArgs, 'usecols' : declarationsFile.getPatColToParseTS()}
            self.df_pat = utils.importChunk(f"{self.interdir}/total_portfolio_activity{addition_to_name}.csv", 250000, **readArgs)

        elif fileID == "patsmp" or fileID == "total_portfolio_activity_sample.csv":
            if select_col:
                readArgs = {**readArgs, 'usecols' : declarationsFile.getPatColToParseTS()}
            self.df_pat = pd.read_csv(f"{self.interdir}/total_portfolio_activity_sample{addition_to_name}.csv", **readArgs)

        elif fileID == 'ltsagg' or fileID == 'linked_ts_aggregated.csv':
            self.df_lts_agg = pd.read_csv(f"{self.interdir}/linked_ts_aggregated.csv", **readArgs)
            self.df_lts_agg = utils.doConvertFromDict(self.df_lts_agg)

        else:
            print("error importing")

    def exportEdited(self, fileID, addition_to_name = ""):
        errorMessage = "No file to export"

        if fileID == "patot" or fileID == "total_portfolio_activity.csv":
            if self.df_pat.empty:
                return print(errorMessage)
            writeArgs = {"index": False}
            utils.exportChunk(self.df_pat, 250000, f"{self.interdir}/total_portfolio_activity{addition_to_name}.csv", **writeArgs)

        if fileID == "cored" or fileID == "df_corporate_details.csv":
            if self.df_corporate_details.empty:
                return print(errorMessage)
            self.df_corporate_details.to_csv(f"{self.interdir}/corporate_details_processed{addition_to_name}.csv", index=False)

        if fileID == "patsmp" or fileID == "total_portfolio_activity_sample{addition_to_name}.csv":
            if self.df_pat_sample.empty:
                return print(errorMessage)
            return print(errorMessage)
            self.df_pat_sample.to_csv(f"{self.interdir}/total_portfolio_activity_sample{addition_to_name}.csv", index=False)

        if fileID == "ltsunc" or fileID == "linked_ts_unconverted.csv":
            if self.df_linked_ts_unc.empty:
                return print(errorMessage)
            writeArgs = {"index": False}
            utils.exportChunk(self.df_linked_ts_unc, 250000, f"{self.interdir}/linked_ts_unconverted{addition_to_name}.csv", **writeArgs)
        else:
            print("error exporting")

        if fileID == "ltsuncsmp" or fileID == "linked_ts_unconverted_sample.csv":
            if self.df_linked_ts_unc_sample.empty:
                return print(errorMessage)
            writeArgs = {"index": False}
            utils.exportChunk(self.df_linked_ts_unc_sample, 250000, f"{self.interdir}/linked_ts_unconverted_sample{addition_to_name}.csv", **writeArgs)

        elif fileID == 'ltsagg' or fileID == 'linked_ts_aggregated.csv':
            writeArgs = {"index": False}
            self.df_lts_agg = self.df_lts_agg(f"{self.interdir}/linked_ts_aggregated.csv", **writeArgs)
        else:
            print("error exporting")



    def portfolioActivitySampler(self, n = 4000, export_after = False, replaceGlobal = False, addition_to_file_name = ""):
        randomizer = np.random.RandomState(self.seed)
        if self.df_pat.empty:
            self.importPortfolioActivity()
        uniqueList = self.df_pat['portfolioid'].unique()
        chosenID = randomizer.choice(uniqueList, n)
        indexID = self.df_pat['portfolioid'].isin(chosenID)
        self.df_pat_sample = self.df_pat[indexID].copy()
        if replaceGlobal:
            self.df_pat = self.df_pat_sample.copy()
        self.exportEdited("patsmp", addition_to_file_name)

    def linked_ts_unconverted_sampler(self, n = 5000, replaceGlobal = False, addition_to_file_name = ""):
        ''''
        Samples observations from self.df_linked_ts. First make sure to import self.df_linked_ts or
        create it with the linkTimeSets functions.
        '''
        randomizer = np.random.RandomState(self.seed)
        if self.df_linked_ts_unc.empty:
            print('No Linked Time file selected to sample')
            return
        unique_personids = self.df_linked_ts_unc['personid'].unique()
        chosenID = randomizer.choice(unique_personids, n)
        self.df_linked_ts_unc_sample  = self.df_linked_ts_unc.query("personid.isin(@chosenID)").copy()
        self.exportEdited("ltsuncsmp", addition_to_file_name)

    def general_sampler(self, data,n = 5000, column_to_sample_from = "personid"):
        ''''
        Samples observations from a dataset and return a dataset which is filtered on these values.
        '''
        randomizer = np.random.RandomState(self.seed)
        unique_ids = data[column_to_sample_from].unique()
        chosenID = randomizer.choice(unique_ids, n)
        data_sample  = self.df_linked_ts_unc.query(f"{column_to_sample_from}.isin(@chosenID)").copy()
        return data_sample

    def doConvertPAT(self):
        if self.df_pat.empty:
            return print("No df_pat to convert")
        self.df_pat = utils.doConvertFromDict(self.df_pat)

    def doConvertLTS(self, do_ignore_errors = True):
        if self.df_linked_ts_unc.empty:
            return print("No df_pat to convert")
        self.df_linked_ts_unc = utils.doConvertFromDict(self.df_linked_ts_unc, ignore_errors=do_ignore_errors)

    def test_in_dataprocessor(self):
        "Method to enable debugging with self. and test the data## "

        test_pivot = pd.pivot_table(self.df_lts_agg, values='personid', index=
        ['has_experian_data', 'has_business_id', 'has_account_overlay'], aggfunc='nunique')

        pass


