import knab_dataprocessor as KD
import utils

# Todo Aparte main gecrëerd zodat knabprocessor ook apart te runnen is.
if __name__ == "__main__":

    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    save_intermediate_results = False # Save the intermediate outputs
    print_information = False # Print things like frequency tables or not
    quarters = True # In which period do we want to aggregate the data?
    cross_sec = False
    time_series = True
    
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
    test.select_ids(quarterly = quarters, subsample = True,
                   sample_size = 500, start_date = "2018", 
                   outname = "base_experian")
    
    #TODO: make subsample variable at the top, also remove subsample
    # from create_base_crosssection function
    
    #----------------MAKE CROSS-SECTIONAL DATASETS-------------------
    if cross_sec:
        df_cross, cross_date = test.create_base_cross_section(date_string="2020-12", 
                            subsample=False, sample_size = 1000,next_period = False, 
                            quarterly=quarters)
        
        
        df_cross_link = test.create_cross_section_perportfolio(df_cross, cross_date, 
                                              outname = "df_cross_portfoliolink",
                                              quarterly= quarters)
        
        test.create_cross_section_perperson(df_cross, df_cross_link,
                                            cross_date, outname = "final_df_quarterly",
                                            quarterly= quarters)
    
    #----------------MAKE TIME SERIES DATASETS-------------------
   
    if time_series:
        test.time_series_from_cross(quarterly = quarters,
                   sample_size = 500, start_date = "2018-01-01")

    #-----------------------------------------------------------------
    
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")

    
    """"
    example of importing a variable and converting the datatypes and selecting columns to preserve memory:
    
    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    
    readArgsPAT = {"usecols": declarationsFile.getPatColToParseTS()}
    datatest = dataProcessor(indirec,interdir,outdirec)
    datatest.importSets("cored")
    datatest.importSets("patsmp",**readArgsPAT)
    datatest.doConvertPAT()
    datatest.linkTimeSets()

    """
