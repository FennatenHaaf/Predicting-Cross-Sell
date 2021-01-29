import pandas as pd
import numpy as np
import os
from datetime import datetime
from time import sleep

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import os
import gc
##Sklearn, statsmodels, linearmodels, scipy

from localMethods import *

useOffline = True
importAll = False
locationProjectFolder = "C:/Users/Ivan Kalinichenka/.AA_Programming/Projects/AA_KnabCase"
locationDataFolder = "C:/Users/Ivan Kalinichenka/.AA_Programming/Projects/AA_KnabCase/rawDataKnab"
locationNewFolder =  "C:/Users/Ivan Kalinichenka/.AA_Programming/Projects/AA_KnabCase/output"

# Can import the contents of this cell when working offline in a particular file. Remember to upload the code you
# created into the workbook!
def changeToProject():
    try:
        global locationProjectFolder
        os.chdir(locationProjectFolder)
    except Exception as e:
        print(e)

def changeToData():
    try:
        global locationDataFolder
        os.chdir(locationDataFolder)
    except Exception as e:
        print(e)

def changeToNewData():
    try:
        global locationNewFolder
        os.chdir(locationNewFolder)
    except Exception as e:
        print(e)

def readData(importString, **readArgs):
    if importString.find(".csv") != -1:
        reader = pd.read_csv
    elif importString.find(".xlsx") != -1 or importString.find(".xls") != -1:
        reader = pd.read_excel
    else:
        print("Unkown variable type")
        return
    return reader(importString, **readArgs)


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

def importFile(fileName, folderCollection = "rawData",chunkSize = 0, **readArgs):
    changeToProject()
    if folderCollection == "rawData":
        changeToData()
    if folderCollection == "newData":
        changeToNewData()
    print("Importing " + fileName)
    if chunkSize > 0:
        imported = importChunk(fileName, chunkSize, **readArgs)
    else:
        imported = readData(fileName, **readArgs)
    changeToProject()
    return imported


def importAndConcat(listOfDataSetNames, folderCollection , chunkSize=0, **readArgs):
    importList = []
    for dataName in listOfDataSetNames:
        importedFile = importFile(dataName, folderCollection=folderCollection, chunkSize=chunkSize, **readArgs)
        importList.append(importedFile)
    return pd.concat(importList, ignore_index=True)


def printUnique(data, nameList):
    for item in nameList:
        text = "Aantal unieke waarden in kolom {} : {}".format(item, str(len(data[item].unique())))
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


# Integrate with prev definition
def mostCommonDict(data, nameList, numberOfValues=1):
    dictOfTables = {}
    for columnName in nameList:
        dictOfTables[columnName] = mostCommon(data, columnName, numberOfValues, True)


def dataDimension(data):
    dimensions = data.shape
    text = "This dataset has {} rows and {} columns".format(dimensions[0], dimensions[1])
    print(text)


def numberOfNaN(data, nameList):
    for item in nameList:
        text = "number of NaN in column {} is {}".format(item, data[item].isna().sum())
        print(text)


def printAll(data):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(data)


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


