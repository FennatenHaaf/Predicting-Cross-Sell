# -*- coding: utf-8 -*-
"""
This file contains extra functions for gaining some insight into the Knab
data 

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd


"""
Printing Intermediate Results
"""
def unique_IDs(df,name):
    """Print number of unique person IDs in a dataset
    """
    nunique = len(pd.DataFrame(df["personid"].unique()))
    print(f'unique person ids in {name}: {nunique}')


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

def recurringValues(data, indexCol, colToTest, threshold = 1):
    def filterMultiple(x):
        return x.count() > threshold
    print(data.groupby(indexCol)[colToTest].filter(filterMultiple).count())

def checkV1(data, value, column="personid"):
    return data[data[column] == value]

def checkAVD(data, column):
    for value in data[column].unique():
        yield data[data[column] == value]

def checkAVL(list):
    for value in list:
        yield value
