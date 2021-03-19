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
from os import path



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
        if self.print_info:
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
            (self.base_df.loc[select,"valid_to_dateeow"]+timedelta(days=7))
                
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
        
        # We want the person IDs for which AT LEAST 1 portfolio has at least
        # 1 activity entry in th period 2018-2020
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
                
                # TODO this is not very efficient, but: make crosssec per person 
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
        
        # We use start and end date
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
                
                
                # Fix some categories for the businessType
                df["businessType"] = df["businessType"].replace("Maatschap", "Maatschap/Stichting")
                df["businessType"] = df["businessType"].replace("Stichting", "Maatschap/Stichting")
                df["businessType"] = df["businessType"].replace("Besloten vennootschap", "Besloten Vennootschap")
                
                #Doe hetzelfde met de Business Type 
                temp = df.loc[:,["personid", "businessType"]]
                indicator =  self.aggregateBusinessPerPerson(temp, 
                                        count_name = "aantal_types")
                df_cross= df_cross.merge(indicator, 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],)

                #### Similar method but this time takes the value of sector of largest portfolio####
                columns_to_use = ["SBIname","SBIcode" ,"SBIsector", "SBIsectorName", "businessType"]
                column_to_agg_on = 'saldototaal'
                merge_list = ["personid"] + columns_to_use + [column_to_agg_on]
                name_filter_list = ['Financiële instellingen (geen verzekeringen en pensioenfondsen)',
                                         'Overige financiële dienstverlening']
                to_merge = self.aggregate_business_to_one_category_per_person(df.loc[:, merge_list].copy(),
                                                                              column_to_agg_on = column_to_agg_on,
                                                                              columns_to_use =  columns_to_use,
                                                                              filter_threshold_high = 0.85,
                                                                              filter_threshold_low = 0.15,
                                                                              filter_string_list = name_filter_list)

                df_cross= df_cross.merge(to_merge,
                              how="left", left_on=["personid"],right_on=["personid"], suffixes = ["", "_on_saldofraction"])

        #Fill NA values for business, joint and retail
        df_cross[['business', 'joint', 'retail']] = df_cross[['business', 'joint', 'retail']].fillna(value = 0)

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
        characteristics.loc[(characteristics["geslacht"]=="Vrouwen"), "geslacht"]="Vrouw"
        
        # Now merge back with df_cross                              
        df_cross= df_cross.merge(characteristics[["personid","birthyear","geslacht"]], 
                                  how="left", left_on=["personid"],
                                  right_on=["personid"],) 
        
        #------------------------ SORT DATASET -------------------------
        df_cross = df_cross.sort_values("personid")   
        df_cross.reset_index(drop=True, inplace=True)
        
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
        # Take the ids and per ID take how often it appears in temp
        indicator = temp["personid"].value_counts().rename_axis(\
                    'personid').to_frame(count_name).reset_index(level=0)
        
        # Merge the data which only appears once and so has only 1 sector type
        IDtemp = indicator["personid"][indicator[count_name]==1]
        temp = temp[temp["personid"].isin(IDtemp)]
        temp= temp.fillna("missing")
        indicator= indicator.merge(temp, 
                          how="left", left_on=["personid"],
                          right_on=["personid"],)
          
        # Next, those who still appear multiple times get type "meerdere"
        indicator = indicator.fillna("meerdere")
        indicator = indicator.replace("missing", np.nan)

        return indicator

    def aggregate_business_to_one_category_per_person(self, data, column_to_agg_on,
                                                      columns_to_use,
                                                      filter_threshold_high = 0.8,
                                                      filter_threshold_low = 0.15,
                                                      filter_string_list = [] ):
        """An alternative to the method above, where only one category is taken
        per person for the business type based on the value"""
        
        if len(filter_string_list) > 0:
            filter_active = True
        else:
            filter_active = False

        for column in columns_to_use:
            tempindex = data[column].notna()
        data = data.drop_duplicates()

        # Calculate the total balance for the person and merge this new value to the data
        balance_total_per_person = pd.DataFrame( data.groupby('personid')[column_to_agg_on].apply(lambda x: x.abs().sum() ) )
        balance_total_per_person.reset_index(inplace = True) #ensure that personid stays in dataset and not index
        data = pd.merge(data, balance_total_per_person, how = "left",
                        on = "personid", suffixes = ["", "_agg"])

        # Create column to create a fraction of total value
        data[f'{column_to_agg_on}_fraction'] = data[column_to_agg_on].abs() / data[f'{column_to_agg_on}_agg'].abs()
        
        # Create a file to merge, start with values which have nan for the column
        # because of 0 division. Create a dataframe for these values and fill 
        # these values with one and grab the first category.
        above_low_threshold_index = data.eval(f"{column_to_agg_on}_fraction >= {filter_threshold_low}")
        tempindex = data.eval(f"{column_to_agg_on}_fraction >= {filter_threshold_high}")

        to_merge = data[~tempindex].query(f"{column_to_agg_on}_fraction != {column_to_agg_on}_fraction").groupby('personid'
                                                                                                     ).first().fillna(1).reset_index()
        to_merge = pd.concat([ data[tempindex], to_merge ], ignore_index =  True)

        # Now merge with values above threshold to create larger merge file
        # Take every person that is not yet in the set to merge
        tempindex = ~(data['personid'].isin(to_merge['personid'])) & above_low_threshold_index
        
        if filter_active:
            # Choose if you want to filter on financial institutions. 
            #Adds extra Values to index on in prev index
            tempindex2 = tempindex
            for column in columns_to_use:
                tempindex2 = tempindex2 & data.loc[tempindex,column].isin(filter_string_list)
            tempindex = tempindex2

        # Concatenate the large file to merge. Here the row with the highest saldo_fraction will be returned
        portfolio_sum_per_sector = data[tempindex].groupby(['personid']).apply(lambda x:
                                                                               x.loc[x[f'{column_to_agg_on}_fraction'].idxmax(),:])
        to_merge = pd.concat([to_merge, portfolio_sum_per_sector], ignore_index = True)

        # Check if there are values below the threshold value after selecting on certain strings. 
        # Return the values from the original data
        templist = to_merge['personid'].to_list()
        portfolio_sum_per_sector2 = data.query("personid != @templist" ).groupby( ['personid'] ).apply( lambda x:x.loc[
            x[f'{column_to_agg_on}_fraction'].idxmax(),:] )
        to_merge = pd.concat([to_merge, portfolio_sum_per_sector2], ignore_index =  True)

        to_merge.drop(f'{column_to_agg_on}_agg', axis = 1, inplace = True)

        return to_merge




# =============================================================================
# Some helper methods that are used to handle time ============================
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
        readArgs = {"usecols": self.getPatColToParseCross(subset = "transactions")}

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
        readArgs = {"usecols": self.getPatColToParseCross(subset = "activity")}
        
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

    ##SELECTED VARIABLES FOR IMPORT
    def getPatColToParseCross(self, subset = "total" ):
        """ Method to select which specific columns are imported from the Knab
        transaction and activity datasets"""

        columnsToParse1 = [
            "dateeow",
            # "yearweek",
            "portfolioid",
            "pakketcategorie"]

        columnsToParse2 = [  # activity
            'overstapserviceyn',
            'betaalalertsyn',
            # 'aantalbetaalalertsubscr',
            # 'aantalbetaalalertsontv',
            'roodstandyn',
            'saldoregulatieyn',
            'appyn',
            'aantalloginsapp',
            'aantalloginsweb',
            'activitystatus'
        ]

        columnsToParse3 = [  # transactions
            'betalenyn',
            'saldobetalen',
            'aantalbetaaltransacties',
            'aantalatmtransacties',
            'aantalpostransacties',
            'aantalfueltransacties',
            'depositoyn',
            'saldodeposito',
            'flexibelsparenyn',
            'saldoflexibelsparen',
            'kwartaalsparenyn',
            'saldokwartaalsparen',
            # De onderstaande variabelen zijn in sommige jaren altjd 0 dus
            # deze gebruiken we niet!
            # 'gemaksbeleggenyn',
            # 'saldogemaksbeleggen',
            # 'participatieyn',
            # 'saldoparticipatie',
            # 'vermogensbeheeryn',
            # 'saldovermogensbeheer',
            'saldototaal',
            'saldolangetermijnsparen',
            'aantaltegenrekeningenlaatsteq'
        ]
        if subset == "total":
            return (columnsToParse1 + columnsToParse2 + columnsToParse3)
        elif subset == "activity":
            return (columnsToParse1 + columnsToParse2)
        elif subset == "transactions":
            return (columnsToParse1 + columnsToParse3)
        elif subset == "all":
            return (columnsToParse1 + columnsToParse2 + columnsToParse3), (columnsToParse1 + columnsToParse2), \
                   (columnsToParse1 + columnsToParse3)
        else:
            print("error getting list")

