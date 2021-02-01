import knab_dataprocessor as KD
import utils

# Todo Aparte main gecrëerd zodat knabprocessor ook apart te runnen is.
if __name__ == "__main__":

    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    save_intermediate_results = False
    
    
    start = utils.get_time()
    
    #----------------INITIALISE DATASET CREATION-------------------
    
    #initialise dataprocessor
    test = KD.dataProcessor(indirec,interdir,outdirec,
                            save_intermediate_results)
    # initialise the base linked data and the base Experian data which are
    # used to create the datasets
    test.link_data() 
    #Create base experian information
    test.create_experian_base()
    
    
    #----------------MAKE CROSS-SECTIONAL DATASETS-------------------
    
    quarters = True # In which period do we want to aggregate the data
    
    df_cross, cross_date, df_next, next_date = test.create_base_cross_section(
        date_string="2020-12", subsample=True, sample_size = 1000, quarterly= quarters)
    
    
    
    end = utils.get_time()
    diff = utils.get_time_diff(start,end)
    print(f"Data creation finished! Total time: {diff}")