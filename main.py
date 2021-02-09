import knab_dataprocessor as KD
import utils

if __name__ == "__main__":

    # Define where our input, output and intermediate data is stored
    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    
    save_intermediate_results = False # Save the intermediate outputs
    print_information = False # Print things like frequency tables or not
    quarters = True # In which period do we want to aggregate the data?
    
    cross_sec = False # Do we want to run the code for getting a single cross-sec
    time_series = True # Do we want to run the code for getting time series data
    
    subsample = True # Do we want to take a subsample 
    sample_size = 500 # The sample size
    start_date = "2018-01-01" # From which moment onwards do we want to use
    # the information in the dataset
    
    # TODO zou ook quarterly een .self variabele kunnnen maken in dataprocessor,
    # alleen dan kan het niet verschillen per dataset maar dat is misschien niet erg
    
    start = utils.get_time()
    
    #----------------INITIALISE DATASET CREATION-------------------
    
    #initialise dataprocessor
    test = KD.dataProcessor(indirec,interdir,outdirec,
                            save_intermediate_results,
                            print_information)
    # initialise the base linked data and the base Experian data which are
    # used to create the datasets
    test.link_data() 
    #Create base experian information and select the ids used -> choose to
    # make subsample here! 
    test.select_ids(quarterly = quarters, subsample = subsample,
                   sample_size = sample_size, start_date = start_date, 
                   outname = "base_experian")
        
    #----------------MAKE CROSS-SECTIONAL DATASETS-------------------
    if cross_sec:
        # Make the base cross-section
        df_cross, cross_date = test.create_base_cross_section(date_string="2020-12", 
                            next_period = False, quarterly=quarters, 
                            outname = "cross_experian")
        # Now aggregate all of the information per portfolio
        df_cross_link = test.create_cross_section_perportfolio(df_cross, cross_date, 
                                              outname = "df_cross_portfoliolink",
                                              quarterly= quarters)
        # Aggregate all of the portfolio information per person ID
        test.create_cross_section_perperson(df_cross, df_cross_link,
                                            cross_date, outname = "final_df_quarterly",
                                            quarterly= quarters)
    
    #----------------MAKE TIME SERIES DATASETS-------------------
    if time_series:

        test.time_series_from_cross(quarterly = quarters, start_date = start_date)
    #-----------------------------------------------------------------
    
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")

    
    
