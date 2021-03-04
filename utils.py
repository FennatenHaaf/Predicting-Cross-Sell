# -*- coding: utf-8 -*-
"""
This file contains extra functions for saving files, etc. 
that could be necessary in other classes

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import numpy as np
import pandas as pd
import os
import shutil
from datetime import datetime
from datetime import timedelta 
from csv import writer
from tqdm import tqdm
import re

import declarationsFile
"""
TIME
"""

def get_time():
    """ This function is to say what time it is at a certain point in
    the simulation
    """
    time = datetime.now()
    time_string = time.strftime("%H:%M:%S")   
    return time_string


def get_time_diff(start, end, time_format= '%H:%M:%S'):
    """ Gets the difference between two time strings""" 
    # convert from string to datetime objects
    start = datetime.strptime(start, time_format)
    end = datetime.strptime(end, time_format)
  
    if start <= end: # e.g., 10:33:26-11:15:49
        return end - start
    else: # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1) # +day
        assert end > start
        return end - start

def select_time_in_data(data, date_column, period_to_use, start, end =""):
    allowed_period_set = {"Q":(r'20[1-2][0,3-9]-?Q[0-4]', '2019Q2'),
                          "M": (r'20[1-2][0,3-9]-?[0-1][0-9]', '201903 or 2019-03'),
                          "D": (r'20[1-2][0,3-9]-?[0-1][0-9]-?[0-3][0-9]', '20190331 or 2019-03-31'),
                         "Y":(r'20[1-2][0,3-9]','2019') }
    assert period_to_use in allowed_period_set, print(f"choose a value from the following set of values {list(allowed_period_set.keys())}")
    assert re.match(allowed_period_set[period_to_use][0], start), print('Wrong format for start period, should be in format '
                                                                     f'{allowed_period_set[period_to_use][1]}')
    if end == "":
        data = data.query(f"{date_column}.dt.to_period('{period_to_use}') >= @start")
    else:
        assert re.match(allowed_period_set[period_to_use][0], end), print('Wrong format for end period')
        data = data.query(f"{date_column}.dt.to_period('{period_to_use}') >= @start &"
                          f" {date_column}.dt.to_period('{period_to_use}') <= @end")
    return data

def infer_date_frequency(date):
    allowed_period_set = {r'20[1-2][0,3-9]-?Q[0-4]':"Q",
                          r'20[1-2][0,3-9]-?[0-1][0-9]':"M",
                          r'20[1-2][0,3-9]-?[0-1][0-9]-?[0-3][0-9]': 'D',
                          r'20[1-2][0,3-9]':"Y"
    }
    for allowed_period in allowed_period_set:
        if re.match(allowed_period,date):
            return allowed_period_set[allowed_period]

    raise print(f"No valid value found for parsed date : {date}")
    

"""
PRINTING
"""   

def printarray(array):
    """Replaces all whitespaces with commas using regular expressions""" 
    
    pattern = re.compile(r'(\d|-)(\s+)(\d|-)')
    arraystring = re.sub(pattern, r'\1, \3', str(array))
    
    return arraystring
   
    

"""
EXPORTING FUNCTIONS
"""
def save_df_to_csv(df, outdir, filename, add_time = True, add_index = False):
    """Saves a dataframe to a csv file in a specified directory, with a
    filename and attached to that the time at which the file is created"""
    
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    
    if add_time:
        today = datetime.now()
        time_string = today.strftime("%Y-%m-%d_%Hh%mm")
  
        df.to_csv(f"{outdir}/{filename}_{time_string}.csv",index=add_index)
    else:
        df.to_csv(f"{outdir}/{filename}.csv",index=add_index)
    

#TODO: Fix this function!
def write_to_csv(data, outdir, filename):
    """Function to write data into an existing csv file"""
    
    print("writing file to csv")
    #print(data)
    
    with open(f"{outdir}/{filename}.csv", 'a') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        data = np.array(data) # Turn into array
        
        for row in tqdm(range(len(np.array(data)))):
            #print(row)
            print(row)
            print(data[row])
            csv_writer.writerow(data[row])

def importChunk(importString, chunksize=500000, **readerArg):
    dfList = []
    for chunk in pd.read_csv(importString, chunksize=chunksize, **readerArg):
        dfList.append(chunk)
    return pd.concat(dfList, ignore_index=True)

def exportChunk(data, chunkSize, exportString, check_if_exists = True ,**writeArgs):
    if os.path.isfile(exportString) and check_if_exists:
        print("File already exists")
        return
    nRows = data.shape[0]
    startIndex = 0
    endIndex = min(startIndex + chunkSize,nRows)
    headerBool = True

    while startIndex < nRows:
        data.iloc[startIndex:endIndex, :].to_csv(path_or_buf=exportString, mode="a",header=headerBool,
                                                 **writeArgs)
        startIndex = endIndex
        endIndex += chunkSize
        endIndex = min(endIndex, nRows)
        headerBool = False

    print("Export of {} completed".format(exportString))

def importAndConcat(listOfDataLocations, chunkSize=0, **readArgs):
    importList = []
    for dataLocation in listOfDataLocations:
        print("Importing from " + dataLocation)
        if chunkSize > 0:
            imported = importChunk(dataLocation, chunkSize, **readArgs)
        else:
            imported = pd.read_csv(dataLocation, **readArgs)
        importList.append(imported)
    return pd.concat(importList, ignore_index=True)

def create_subfolder_and_import_files(first_date, last_date, subfolder = "", folder_name_addition = "", find_list = [] ):
    """
    Creates a subfolder for a certain time period that has been processed.
    -Need to fill in a first_date, last_date.
    -Also possible to go to a subfolder or add an additional stirng to the folder name
    First checks if the folder exists to backup previously saved files.
    """
    if not subfolder == "":
        subfolder_path = f"{subfolder}/"
    else:
        subfolder_path = ""

    folder_string = f"{subfolder_path}{first_date}_{last_date}{folder_name_addition}"
    if os.path.exists(folder_string):
        print('Folder already exists, proceeding to backup old folder')
        time_string = datetime.now().strftime("M%mD%d_%H%M")
        try:
            os.mkdir("z_old")
        except:
            pass
        new_folder_string = f"{subfolder_path}/z_old/{first_date}_{last_date}{folder_name_addition}_{time_string}"
        os.mkdir(new_folder_string)
        for item in os.listdir(folder_string):
            shutil.copy2(f"{folder_string}/{item}", new_folder_string)
        shutil.rmtree(folder_string)

    os.mkdir(folder_string)
    print("New folder created")
    for item in os.listdir(subfolder_path):
        for string_to_find in find_list:
            if string_to_find in item:
                shutil.copy2(f"{subfolder_path}{item}", f"{folder_string}/{item}")
                break
    print("Files have been copied to new Folder")

def replace_time_period_folder(first_date = "", last_date= "", subfolder = None, remove_list = [] ):
    "Replaces all the files for the final_df with another time period"
    if subfolder != None:
        subfolder = f"{subfolder}/"
    os.listdir(subfolder)
    for item in os.listdir(subfolder):
        for string_to_delete in remove_list:
            if string_to_delete in item:
                os.remove(f"{subfolder}{item}")
                break

    if not first_date == "":
        new_folder = f"{subfolder}{first_date}_{last_date}"
        for item in os.listdir(new_folder):
            shutil.copy2(f"{new_folder}/{item}", f"{subfolder}")


"""
Large Data Methods
"""
def selectChunk(data, numberOfChunks=4):
    totalRows = data.shape[0]
    chunkSize = totalRows // numberOfChunks
    prevValue = 0
    chunkList = []
    for i in range(1, numberOfChunks):
        currentValue = chunkSize * i
        chunk = data.iloc[prevValue:currentValue, :]
        chunkList.append(chunk)
        prevValue += 11

    if (totalRows - currentValue) > 0:
        chunk = data.iloc[currentValue:totalRows, :]
        chunkList.append(chunk)

    return pd.concat(chunkList, ignore_index=True)

def doConvertFromDict(data, ignore_errors = True, exclusion_list = []):
    if ignore_errors == True:
        error_handling = 'ignore'
    else:
        error_handling = 'raise'
    fullDict = declarationsFile.getConvertDict()
    endDict = doDictIntersect(small_set=data.columns, full_dict=fullDict, exclusion_list = exclusion_list)
    return data.astype(endDict, errors = error_handling)

def doConvertNumeric(self, data):
    for item in data.dtypes.index:
        currentDtype = str(data.dtypes[item]).lower()
        if "float" in currentDtype or "int" in currentDtype:
            data.loc[:, item] = pd.to_numeric(data.loc[:, item], downcast="integer")
    return data

def doDictIntersect(small_set, full_dict : dict, exclusion_list = []):
    end_dict = {}
    resulting_set = (set(small_set) & set(full_dict)) - set(exclusion_list)
    for value in resulting_set:
        end_dict[value] = full_dict[value]
    return end_dict

def doListIntersect(list_to_check, full_list : list, exclusion_list = []):
    result = ( set(list_to_check) & set(full_list) ) - set(exclusion_list)
    return list(result)

def do_find_and_select_from_list(list_to_search,search_string_list, exclusion_string_list = []):
    variable_list = []
    for variable in list_to_search:
        for search_string in search_string_list:
            if search_string in variable:
                append_if_found = True
                for excluded_string in exclusion_string_list:
                    if excluded_string in variable:
                        append_if_found = False
                        break
                if append_if_found:
                    variable_list.append(variable)
    return variable_list

def check_and_drop_columns(data: pd.DataFrame, columns_to_drop):
    drop_list = doListIntersect(data.columns, columns_to_drop)
    data.drop(drop_list, inplace = True)
    return data

def check_and_select_columns(data: pd.DataFrame, columns_to_select):
    select_list = doListIntersect(columns_to_select, data.columns)
    return data