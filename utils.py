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
from datetime import datetime
from datetime import timedelta 
from csv import writer
from tqdm import tqdm


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
    

#TODO: Fix this function
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

def exportChunk(data, chunkSize, exportString, **writeArgs):
    if os.path.isfile(exportString):
        print("File already exists")
        return
    nRows = data.shape[0]
    if chunkSize > nRows:
        chunkSize = nRows
    data.iloc[:chunkSize, :].to_csv(path_or_buf=exportString, mode="a", **writeArgs)

    startIndex = chunkSize
    endIndex = startIndex + chunkSize
    while endIndex <= nRows:
        data.iloc[startIndex:endIndex, :].to_csv(path_or_buf=exportString, mode="a", header=False,
                                                 **writeArgs)
        startIndex += chunkSize
        endIndex += chunkSize

    restChunk = nRows - startIndex
    if restChunk > 0:
        data.iloc[startIndex:nRows, :].to_csv(path_or_buf=exportString, mode="a", header=False, **writeArgs)
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

"""
Printing Intermediate Results
"""
def numberOfNaN(data, nameList):
    for item in nameList:
        text = "number of NaN in column {} is {}".format(item, data[item].isna().sum())
        print(text)

def mostCommon(data, columnName, numberOfValues=1, returnFormat=False, returnVals=False, alphabetSort=False,
               noPrint=False):
    '''
    Returns the most common values that can be found with the frequency and fraction
    data | should be a pandas Dataframe
    columnName | Name of the column to return, must be one column
    numberOfValues | is the number of values to show. If larger than number of available values, retrn smaller table
    returnFormat | Returns a formatted table and prints. IF returnVals is also True, will override this variable
    returnVals | Standard False, will not print but only return a full frequency and fraction table.
    '''
    table = data.value_counts(subset=columnName, sort=True)
    table2 = data.value_counts(subset=columnName, sort=True, normalize=True)
    tableSum = table.sum()
    tableString, tableString2 = "Frequency", "Fraction"
    table = table.to_frame(tableString)
    table2 = table2.to_frame(tableString2)
    table = pd.concat([table, table2], axis=1)

    if not returnVals:
        print("\n Frequency Table for {}".format(columnName))

        def lastStep():
            # nonlocal table
            # nonlocal tableString
            # nonlocal noPrint
            table.loc["Total", :] = [tableSum, 1]
            table[tableString] = pd.to_numeric(table[tableString], downcast="unsigned")
            if not noPrint:
                print(table)
                print("\n")

        if (table.shape[0] - 2) <= numberOfValues:  # -2 because rubric other will else be equal to that category
            if alphabetSort:
                table.sort_index(inplace=True)
            lastStep()
        else:
            table = table.iloc[:numberOfValues]
            freqSum, fracSum = table.sum()
            if alphabetSort:
                table.sort_index(inplace=True)
            table.loc["Other", :] = [tableSum - freqSum, 1 - fracSum]
            lastStep()
        print("\n")
    if returnFormat or returnVals:
        return table

def mostCommonDict(data, nameList, numberOfValues=1):
    dictOfTables = {}
    for columnName in nameList:
        dictOfTables[columnName] = mostCommon(data, columnName, numberOfValues, True)
"""
Formatting Functions
"""