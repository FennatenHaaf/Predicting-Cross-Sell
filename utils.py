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

def get_datetime():
    datetime_string = datetime.now().strftime("%m%d_%H%M")
    return datetime_string


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
    """
    Selecting a slice of data for a certain time. Will check if the date is correct
    :param data: data to slice
    :param date_column: column containing the date
    :param period_to_use: period to use
    :param start: starting period
    :param end: ending period
    :return:
    """
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
    """
    date frequency to return for conversion
    :param date: date to put into
    :return:
    """
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

def printarray(array, removenewlines = False):
    """
    Replaces all whitespaces with commas using regular expressions
    :param array:
    :param removenewlines:
    :return:
    """
    
    if removenewlines:
        string = re.sub(r"\r\n", " ", str(array))
    else: 
        string  = str(array)
        
    pattern = re.compile(r'(\d|-|\])(\s+)(\d|-|\[)')
    arraystring = re.sub(pattern, r'\1, \3', str(array))
    
    return arraystring
   
def print_seperated_list(a_list, seperator_char = "|"):
    """
    Returns a string with all items in a list printed in one string
    and seperated by a seperator_char (default = '|').
    :param a_list:
    :param seperator_char:
    :return:
    """
    return_string = f" {seperator_char} "
    for item in a_list:
        return_string = f"{return_string} {str(item)} |"
    return return_string


"""
EXPORTING FUNCTIONS
"""
def save_df_to_csv(df, outdir, filename, add_time = True, add_index = False):
    """
    Saves a dataframe to a csv file in a specified directory, with a
    filename and attached to that the time at which the file is created
    :param df:
    :param outdir:
    :param filename:
    :param add_time:
    :param add_index:
    :return:
    """
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
    """
    Function to write data into an existing csv file
    :param data:
    :param outdir:
    :param filename:
    :return:
    """
    
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

def create_result_archive(subfolder = None, archive_name = "archive",subarchive_addition = None,
                          files_string_to_archive_list = [], file_string_to_exclude_list = []):
    """
    Creates a folder to create an archive for different results.
    :param subfolder:
    :param archive_name:
    :param subarchive_addition:
    :param files_string_to_archive_list:
    :param file_string_to_exclude_list:
    :return:
    """

    if subfolder != None:
        subfolder = f"{subfolder}/"

    archive_folder_name = f"{subfolder}{archive_name}/"
    if not os.path.exists(archive_folder_name):
        os.mkdir(archive_folder_name)

    if subarchive_addition != None:
        subarchive_folder_path = f"{archive_folder_name}{archive_name}{subarchive_addition}/"
        if not os.path.exists(subarchive_folder_path):
            os.mkdir(subarchive_folder_path)
    else:
        subarchive_folder_path = archive_folder_name

    time_string = datetime.now().strftime("%m%d_%H%M%S")
    for item in os.listdir(subfolder):
        not_include = False
        for include_string in files_string_to_archive_list:
            if not_include:
                continue

            if include_string in item:
                for exclude_string in file_string_to_exclude_list:
                    if exclude_string in item:
                        not_include = True
                        break
                if not_include:
                    continue
                else:
                    shutil.copy2(f"{subfolder}{item}", f"{subarchive_folder_path}{time_string}_{item}")
                break

def create_subfolder_and_import_files(first_date, last_date, subfolder = "", folder_name_addition = "", find_list = [] ):
    """
    Creates a subfolder for a certain time period that has been processed.
    -Need to fill in a first_date, last_date.
    -Also possible to go to a subfolder or add an additional stirng to the folder name
    First checks if the folder exists to backup previously saved files.
    :param first_date: first_date of the data or the date of the data
    :param last_date: last_date to use
    :param subfolder: Name of the subfolder within the larger folder where to create a new subfolder
    :param folder_name_addition: addition to the subfolder name to archive
    :param find_list: names of files that should be found to be copied to this archive
    :return:
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
    """
    Replaces all the files with a string that is contained in the remove list with another time period
    :param first_date:
    :param last_date:
    :param subfolder:
    :param remove_list:
    :return:
    """
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

def importChunk(importString, chunksize=500000, **readerArg):
    """
    Importing chunks of data when the data to  be imported is too large
    to fit into the memory directly
    :param importString: filename to import
    :param chunksize: the size of the chunks used to import in number of rows
    :param readerArg: arguments to be parsed in pd.read_csv
    :return:
    """
    dfList = []
    for chunk in pd.read_csv(importString, chunksize=chunksize, **readerArg):
        dfList.append(chunk)
    return pd.concat(dfList, ignore_index=True)

def exportChunk(data, chunkSize, exportString, check_if_exists = True ,**writeArgs):
    """
    Exporting data dat is too large to fit into memory immediately
    :param data:
    :param chunkSize:
    :param exportString:
    :param check_if_exists:
    :param writeArgs:
    :return:
    """
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
    """
    Import files and concatenate immediately to a new file
    :param listOfDataLocations:
    :param chunkSize:
    :param readArgs:
    :return:
    """
    importList = []
    for dataLocation in listOfDataLocations:
        print("Importing from " + dataLocation)
        if chunkSize > 0:
            imported = importChunk(dataLocation, chunkSize, **readArgs)
        else:
            imported = pd.read_csv(dataLocation, **readArgs)
        importList.append(imported)
    return pd.concat(importList, ignore_index=True)


"""
Large Data Methods
"""
def doConvertFromDict(data, ignore_errors = True, exclusion_list = []):
    if ignore_errors == True:
        error_handling = 'ignore'
    else:
        error_handling = 'raise'
    fullDict = getConvertDict()
    endDict = doDictIntersect(small_set=data.columns, full_dict=fullDict, exclusion_list = exclusion_list)
    return data.astype(endDict, errors = error_handling)

def doConvertNumeric(self, data):
    for item in data.dtypes.index:
        currentDtype = str(data.dtypes[item]).lower()
        if "float" in currentDtype or "int" in currentDtype:
            data.loc[:, item] = pd.to_numeric(data.loc[:, item], downcast="integer")
    return data

def doDictIntersect(small_set, full_dict : dict, exclusion_list = []):
    """
    :param small_set: smaller dictionary or set
    :param full_dict: larger list to search if the smaller set has been contained
    :param exclusion_list: values not to include
    :return:
    """
    end_dict = {}
    resulting_set = (set(small_set) & set(full_dict)) - set(exclusion_list)
    for value in resulting_set:
        end_dict[value] = full_dict[value]
    return end_dict

def doListIntersect(list_to_check, full_list : list, exclusion_list = []):
    """
    :param list_to_check: smaller list or list of strings that should be found in full list
    :param full_list: larger list to check
    :param exclusion_list: full names of strings that should be excluded from the intersection
    """
    result = ( set(list_to_check) & set(full_list) ) - set(exclusion_list)
    return list(result)

def do_find_and_select_from_list(list_to_search: list,search_string_list: list, exclusion_string_list = []):
    """
    :param list_to_search: input a list that has to be searched
    :param search_string_list: if one string in the list contains a string from the search list, will return unless
    this string is also in the exclusion list
    :param exclusion_string_list: strings to be excluded when found in the list of string to search
    :return:
    """
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


def getConvertDict():
    """
    Conversion of different variables used through the analysis in order to save memory
    at different points. Outputs a large dictionary to use as input for the doConvertFromDict() function
    """

    datatypeGeneralActivity = {
        'dateeow'        : "datetime64",
        'yearweek'       : "string",
        'portfolioid'    : "category",
        'pakketcategorie': "category"}

    datatypeActivity = {
        'overstapserviceyn'      : "uint8",
        'betaalalertsyn'         : "uint8",
        'aantalbetaalalertsubscr': "uint16",
        'aantalbetaalalertsontv' : "uint16",
        'roodstandyn'            : "uint8",
        'saldoregulatieyn'       : "uint8",
        'appyn'                  : "uint8",
        'aantalloginsapp'        : "uint16",
        'aantalloginsweb'        : "uint16",
        'activitystatus'         : "category"}

    datatypeTrans = {
        'betalenyn'                    : "uint8",
        'saldobetalen'                 : "int32",
        'aantalbetaaltransacties'      : "uint16",
        'aantalatmtransacties'         : "uint16",
        'aantalpostransacties'         : "uint16",
        'aantalfueltransacties'        : "uint16",
        'depositoyn'                   : "uint8",
        'saldodeposito'                : "uint32",
        'flexibelsparenyn'             : "uint8",
        'saldoflexibelsparen'          : "uint32",
        'kwartaalsparenyn'             : "uint8",
        'saldokwartaalsparen'          : "uint32",
        'gemaksbeleggenyn'             : "uint8",
        'saldogemaksbeleggen'          : "uint32",
        'participatieyn'               : "uint8",
        'saldoparticipatie'            : "uint32",
        'vermogensbeheeryn'            : "uint8",
        'saldovermogensbeheer'         : "uint32",
        'saldototaal'                  : "int32",
        'saldolangetermijnsparen'      : "uint32",
        'aantaltegenrekeningenlaatsteq': "uint16"}

    df_lpp_dict = {
        "validfromdate"      : "datetime64",
        "validfromyearweek"  : "int32",
        "personid"           : "category",
        "iscorporatepersonyn": "uint8"
    }

    df_exp_dict = {
        'valid_from_dateeow'  : "datetime64",
        'valid_to_dateeow'    : "datetime64",
        'age_hh'              : "uint8",
        'hh_child'            : "uint8",
        'hh_size'             : "uint8",
        'income'              : "uint8",
        'educat4'             : "uint8",
        'housetype'           : "uint8",
        'finergy_tp'          : "category",
        'lfase'               : "uint8",
        'business'            : "uint8",
        'huidigewaarde_klasse': "uint8"}

    df_cor_dict = {
        'personid'           : "category",
        'businessid'         : 'category',
        'businessType'       : "category",
        'foundingDate'       : 'uint16',
        'businessAgeInDays'  : 'uint16',
        'businessAgeInMonths': 'uint16',
        'businessAgeInYears' : 'float16',
        'foundingYear'       : 'uint16',
        'SBIcode'            : 'category',
        'SBIname'            : 'category',
        'SBIsector'          : 'category',
        'SBIsectorName'      : 'category'
    }

    df_pin_dict = {
        'dateinstroomweek': 'datetime64[ns]',
        'instroomjaarweek': 'datetime64[ns]',
        'instroompakket'  : 'category',
        'birthyear'       : 'uint16',
        'geslacht'        : 'category',
        'type'            : 'category',
        'enofyn'          : 'uint8'
    }

    df_bhk_dict = {
        'boekhoudkoppeling': 'category'
    }

    time_sets_new = {
        'has_experian_data'                 : 'uint8',
        'has_business_id'                   : 'uint8',
        'has_account_overlay'               : 'uint8',
        'indicator_corp_and_retail'         : 'uint8',
        'indicator_corp_and_retail_business': 'uint8',
        'iscorporatepersonyn_business'      : 'uint8',
        'business_id_with_corp_and_retail'  : 'uint8',
        'retail_id_with_corp_and_retail'    : 'uint8'
    }
    time_set_new2 = {
        'converted_period'         : 'datetime64',
        'indicator_1_inactief'     : 'uint8',
        'indicator_2_SparenOnlyYN' : 'uint8',
        'indicator_3_actief'       : 'uint8',
        'indicator_4_primaire bank': 'uint8'
    }

    final_df_dict = {
        'aantal_SBI'                            : 'uint8',
        'aantal_sector'                         : 'uint8',
        'aantal_types'                          : 'uint8',
        'aantalatmtransacties_business'         : 'uint8',
        'aantalatmtransacties_joint'            : 'uint8',
        'aantalatmtransacties_retail'           : 'uint16',
        'aantalbetaaltransacties_business'      : 'uint16',
        'aantalbetaaltransacties_joint'         : 'uint16',
        'aantalbetaaltransacties_retail'        : 'uint8',
        'aantalfueltransacties_business'        : 'uint8',
        'aantalfueltransacties_joint'           : 'uint8',
        'aantalfueltransacties_retail'          : 'uint8',
        'aantalloginsapp_business'              : 'uint16',
        'aantalloginsapp_joint'                 : 'uint16',
        'aantalloginsapp_retail'                : 'uint16',
        'aantalloginsweb_business'              : 'uint16',
        'aantalloginsweb_joint'                 : 'uint16',
        'aantalloginsweb_retail'                : 'uint16',
        'aantalpostransacties_business'         : 'uint16',
        'aantalpostransacties_joint'            : 'uint16',
        'aantalpostransacties_retail'           : 'uint16',
        'aantaltegenrekeningenlaatsteq_business': 'uint16',
        'aantaltegenrekeningenlaatsteq_joint'   : 'uint16',
        'aantaltegenrekeningenlaatsteq_retail'  : 'uint16',
        'accountoverlay'                        : 'uint8',
        'activitystatus_business'               : 'uint8',
        'activitystatus_joint'                  : 'uint8',
        'activitystatus_retail'                 : 'uint8',
        'betalenyn_business'                    : 'uint8',
        'betalenyn_joint'                       : 'uint8',
        'betalenyn_retail'                      : 'uint8',
        'depositoyn_business'                   : 'uint8',
        'depositoyn_joint'                      : 'uint8',
        'depositoyn_retail'                     : 'uint8',
        'flexibelsparenyn_business'             : 'uint8',
        'flexibelsparenyn_joint'                : 'uint8',
        'flexibelsparenyn_retail'               : 'uint8',
        'geslacht_joint'                        : 'category',
        'joint'                                 : 'uint8',
        'kwartaalsparenyn_business'             : "uint8",
        'kwartaalsparenyn_joint'                : "uint8",
        'kwartaalsparenyn_retail'               : "uint8",
        'retail'                                : "uint8",
        'saldototaal_business'                  : 'int32',
        'saldototaal_joint'                     : 'int32',
        'saldototaal_retail'                    : 'int32'
    }

    final_df_dict2 = {'SBIname_on_saldofraction'      : "category",
                      'SBIsectorName_on_saldofraction': "category",
                      'businessType_on_saldofraction' : "category",
                      'saldototaal_agg'               : 'int32',
                      'saldototaal_fraction'          : 'float8',
                      'SBIcode_on_saldofraction'      : "uint8",
                      'SBIsector_on_saldofraction'    : "category",
                      'period_obs'                    : "datetime64"
                      }

    lowercase_input_df_dict = {
        'aantal_sbi'                    : "uint8",
        'businessageinyears'            : "float16",
        'businesstype'                  : "category",
        'businesstype_on_saldofraction' : "category",
        'sbicode'                       : "uint8",
        'sbicode_on_saldofraction'      : "uint8",
        'sbiname'                       : "category",
        'sbiname_on_saldofraction'      : "category",
        'sbisector'                     : "category",
        'sbisector_on_saldofraction'    : "category",
        'sbisectorname'                 : "category",
        'sbisectorname_on_saldofraction': "category"}

    additional_dataprocess_dict = {
        'aantalproducten_totaal'           : "uint8",
        'aantalproducten_totaal_business'  : "uint8",
        'aantalproducten_totaal_joint'     : "uint8",
        'aantalproducten_totaal_retail'    : "uint8",
        'aantaltransacties_totaal'         : "uint8",
        'aantaltransacties_totaal_business': "uint16",
        'aantaltransacties_totaal_joint'   : "uint16",
        'aantaltransacties_totaal_retail'  : "uint16",
        'logins_totaal'                    : "uint8",
        'logins_totaal_business'           : "uint8",
        'logins_totaal_joint'              : "uint8",
        'logins_totaal_retail'             : "uint8"
    }

    return {**datatypeGeneralActivity, **datatypeActivity, **datatypeTrans, **df_lpp_dict, **df_exp_dict,
            **df_cor_dict, **df_pin_dict, **time_sets_new, **time_set_new2, **final_df_dict, **final_df_dict2,
            **lowercase_input_df_dict, **additional_dataprocess_dict}