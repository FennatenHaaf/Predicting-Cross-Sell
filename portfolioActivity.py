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

## Importing File
from generalMethodsClasses import *

def dropColumn(data, columnsToDrop = []):
    try:
        data.drop(columnsToDrop, inplace = True)
    except:
        print("No column to drop selected or not existant")
    return data

def importPortfolioActivity(convertData = False, selecTcolumns = False,**readArgs):
    if convertData:
        datatypeConvertAll = getPatConvert()
    else:
        datatypeConvertAll = {}

    if selecTcolumns:
        readArgs = {**readArgs, "usecols":getPatColToParse()}
    else:
        pass
    rawDataCollection = "rawData"
    changeToData()

    tempList = ["portfolio_activity_business_2018.csv", "portfolio_activity_business_2019.csv",
                "portfolio_activity_business_2020.csv"]
    pab1820 = importAndConcat(tempList, folderCollection=rawDataCollection, **readArgs)

    print(pab1820.shape," are the dimensions of pab18-20")

    tempList = ["portfolio_activity_retail_2018.csv", "portfolio_activity_retail_2019.csv",
                "portfolio_activity_retail_2020.csv"]
    par1820 = importAndConcat(tempList, folderCollection=rawDataCollection,**readArgs)
    print(par1820.shape, " are the dimensions of par18-20")

    pa1820 = pd.concat(
        [pab1820, par1820], ignore_index=True)
    del par1820, pab1820
    gc.collect()
    print(pa1820.shape, " are the dimensions of pa 18-20")


    tempList = ["portfolio_activity_transactions_business_2018.csv",
                "portfolio_activity_transactions_business_2019.csv",
                "portfolio_activity_transactions_business_2020.csv"]
    patb1820 = importAndConcat(tempList, folderCollection=rawDataCollection,**readArgs)
    print(patb1820.shape, " are the dimensions of patb 18-20")

    tempList = ["portfolio_activity_transactions_retail_2018.csv", "portfolio_activity_transactions_retail_2019.csv",
                "portfolio_activity_transactions_retail_2020.csv"]
    patr1820 = importAndConcat(tempList, folderCollection=rawDataCollection,**readArgs)
    print(patr1820.shape, " are the dimensions of patr 18-20")

    pat1820 = pd.concat(
        [patr1820, patb1820],
        ignore_index=True)
    print(pat1820.shape, " are the dimensions of pa before merge 18-20")
    del patr1820, patb1820
    gc.collect()

    pat1820 = pd.merge(pa1820,
                                                  pat1820, how="inner",
                                                  on=["dateeow", "yearweek", "portfolioid", "pakketcategorie"])
    del pa1820
    gc.collect()
    print(pat1820.shape, " are the dimensions of pa 18-20")

    pa1820 = pat1820.astype(datatypeConvertAll)

    tempList = ["portfolio_activity_business.csv", "portfolio_activity_retail.csv", ]
    pa1420 = importAndConcat(tempList, folderCollection=rawDataCollection, chunkSize=500000,**readArgs)
    print(pa1420.shape, " are the dimensions of pa before merge 14-20")

    tempList = ["portfolio_activity_transaction_business.csv", "portfolio_activity_transaction_retail.csv"]
    pat1420 = importAndConcat(tempList, folderCollection=rawDataCollection,
                                                                chunkSize=500000, **readArgs)
    print(pat1420.shape, " are the dimensions of pa before merge 14-20")
    patotal1420 = pd.merge(pa1420,
                                                  pat1420, how="inner",
                                                  on=["dateeow", "yearweek", "portfolioid", "pakketcategorie"])
    del pa1420,pat1420
    gc.collect()
    print(pat1420.shape, " are the dimensions of pat 14-20")

    patotal1420 = patotal1420.astype(datatypeConvertAll)

    pat = pd.concat([patotal1420,pat1820])
    print(pat.shape, " are the dimensions of pat 14-20")
    return pat


def explorePA(data):
    datatypeConvertAll = getPatConvert()

    # readArgs = {"usecols":columnsToParse}
    changeToNewData()
    pat = importChunk("total_portfolio_activity_larger.csv", chunksize=250000)
    # pat = pd.read_csv("total_portfolio_activity_larger_sample.csv")
    changeToProject()

    patSubID = ["dateeow", "yearweek", "portfolioid"]
    patSubID1 = patSubID + ['pakketcategorie',
                            'overstapserviceyn', 'betaalalertsyn', 'aantalbetaalalertsubscr',
                            'aantalbetaalalertsontv', 'roodstandyn', 'saldoregulatieyn', 'appyn',
                            'aantalloginsapp', 'aantalloginsweb', 'activitystatus']

    patSubID2 = patSubID + ['betalenyn',
                            'saldobetalen', 'aantalbetaaltransacties', 'aantalatmtransacties',
                            'aantalpostransacties', 'aantalfueltransacties', 'aantaltegenrekeningenlaatsteq']

    patSubID3 = patSubID + ['betalenyn',
                            'saldobetalen', 'depositoyn',
                            'saldodeposito', 'flexibelsparenyn', 'saldoflexibelsparen',
                            'kwartaalsparenyn', 'saldokwartaalsparen', 'gemaksbeleggenyn',
                            'saldogemaksbeleggen', 'participatieyn', 'saldoparticipatie',
                            'vermogensbeheeryn', 'saldovermogensbeheer', 'saldototaal',
                            'saldolangetermijnsparen']

    patSub1 = pat.loc[:, patSubID1]
    patSub2 = pat.loc[:, patSubID2]
    patSub3 = pat.loc[:, patSubID3]

    print(patSub1["pakketcategorie"].unique().tolist())
    patSub1["indicatorZP"] = 0
    patSub1["indicatorPB"] = 0
    patSub1["indicatorKB"] = 0

    patSub1.loc[patSub1["pakketcategorie"] == "Zakelijk pakket", "indicatorZP"] = 1
    patSub1.loc[patSub1["pakketcategorie"] == "Particulier Betalend", "indicatorPB"] = 1
    patSub1.loc[patSub1["pakketcategorie"] == "Knab Basis", "indicatorKB"] = 1

    # Volgende stap: Indicator variabele voor deze pakketen maken en kijken hoeveel mensen van pakket wisselen
    patS1gr = patSub1.groupby("portfolioid")[["indicatorZP", "indicatorPB", "indicatorKB"]].max()
    patS1gr["multiple"] = patS1gr["indicatorZP"] + patS1gr["indicatorPB"] + patS1gr["indicatorKB"]
    pats1Multi = patS1gr[patS1gr["multiple"] > 1]
    text = "Van de {} mensen heeft {} meerdere portfolio's.".format(patS1gr.shape[0], pats1Multi.shape[0])
    print(text)
    res1 = pd.eval("(pats1Multi['indicatorZP'] == 1) & (pats1Multi['multiple'] > 1) ")
    res2 = pd.eval("(pats1Multi['indicatorKB'] == 1) & (pats1Multi['indicatorPB'] == 1) ")
    print("met Zakelijk en particulier: ", res1.sum(), "alleen particulier: ", res2.sum())

    businessAndRetailList = pats1Multi[res1].index.to_list()
    businessAndRetailObservations = pat[pat["portfolioid"].isin(businessAndRetailList)].copy()
    businessAndRetailObservations.sort_values(["portfolioid", "dateeow"], inplace=True)

def transformPA(pat):
    columnsToParse = getPatColToParse()
    datatypeConvertAll = getPatConvert()

    pat = pat.astype(datatypeConvertAll)
    pat["yearQuarter"] = pat["dateeow"].dt.to_period("Q")

    ##Convert to pivot
    patcolumns = ['dateeow', 'saldobetalen',
                  'aantalloginsapp', 'aantalloginsweb', 'betalenyn']
    pataggfunc = {
        'dateeow': min,
        'saldobetalen': "mean",
        'aantalloginsapp': sum,
        'aantalloginsweb': sum,
        "betalenyn": max}

    indexColumns = ["portfolioid", "yearMonth"]

    patpivot = pd.pivot_table(pat, values=patcolumns, index=indexColumns, aggfunc=pataggfunc)
    patpivot.dropna(inplace=True)

    return pat


##Dictionaries for transforming
def getPatConvert(full = False):
    datatypeConvertP1 = {
        'dateeow': "datetime64",
        'yearweek': "string",
        'portfolioid': "category",
        'pakketcategorie': "category"}

    datatypeConvertP2 = {
        **datatypeConvertP1 ,
        'overstapserviceyn': "uint8",
        # 'betaalalertsyn': "uint8",
        # 'aantalbetaalalertsubscr': "uint16",
        # 'aantalbetaalalertsontv': "uint16",
        'roodstandyn': "uint8",
        'saldoregulatieyn': "uint8",
        'appyn': "uint8",
        'aantalloginsapp': "uint16",
        'aantalloginsweb': "uint16",
        'activitystatus': "category"}

    datatypeConvertP3 = {
        **datatypeConvertP1 ,
        'betalenyn': "uint8",
        'saldobetalen': "int32",
        'aantalbetaaltransacties': "uint16",
        'aantalatmtransacties': "uint16",
        'aantalpostransacties': "uint16",
        'aantalfueltransacties': "uint16",
        'depositoyn': "uint8",
        'saldodeposito': "uint32",
        'flexibelsparenyn': "uint8",
        'saldoflexibelsparen': "uint32",
        'kwartaalsparenyn': "uint8",
        'saldokwartaalsparen': "uint32",
        'gemaksbeleggenyn': "uint8",
        'saldogemaksbeleggen': "uint32",
        'participatieyn': "uint8",
        'saldoparticipatie': "uint32",
        'vermogensbeheeryn': "uint8",
        'saldovermogensbeheer': "uint32",
        'saldototaal': "int32",
        'saldolangetermijnsparen': "uint32",
        'aantaltegenrekeningenlaatsteq': "uint16"}

    datatypeConvertAll = {**datatypeConvertP2, **datatypeConvertP3}

    if full:
        return datatypeConvertAll, datatypeConvertP2, datatypeConvertP3
    else:
        return datatypeConvertAll


def getPatColToParse(full = False):
    columnsToParse1 = [
        "dateeow",
        "yearweek",
        "portfolioid",
        "pakketcategorie"]

    columnsToParse2 = [
        'overstapserviceyn',
        # 'betaalalertsyn',
        # 'aantalbetaalalertsubscr',
        # 'aantalbetaalalertsontv',
        'roodstandyn',
        'saldoregulatieyn',
        'appyn',
        'aantalloginsapp',
        'aantalloginsweb',
        'activitystatus'
    ]

    columnsToParse3 = [
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
        'gemaksbeleggenyn',
        'saldogemaksbeleggen',
        'participatieyn',
        'saldoparticipatie',
        'vermogensbeheeryn',
        'saldovermogensbeheer',
        'saldototaal',
        'saldolangetermijnsparen',
        'aantaltegenrekeningenlaatsteq'
    ]

    if full:
        return (columnsToParse1+columnsToParse2+columnsToParse3), (columnsToParse1+columnsToParse2), \
               (columnsToParse1+columnsToParse3)
    else:
        return (columnsToParse1 + columnsToParse2 + columnsToParse3)

# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html?highlight=basics#dtypes

##Conversie via pivot
def getAggFuncDict():
    aggFuncDictP1 = {
    'dateeow': min}

    aggFuncDictP2 = {
        **aggFuncDictP1,
        'overstapserviceyn': "mean",
        # 'betaalalertsyn': "mean",
        # 'aantalbetaalalertsubscr': "uint16",
        # 'aantalbetaalalertsontv': "uint16",
        'roodstandyn': "mean",
        'saldoregulatieyn': "mean",
        'appyn': "mean",
        'aantalloginsapp': sum,
        'aantalloginsweb': sum
        # 'activitystatus': "category",
        }

    aggFuncDictP3 = {
        **aggFuncDictP1,
        'betalenyn': "uint8",
        'saldobetalen': "int32",
        'aantalbetaaltransacties': "uint16",
        'aantalatmtransacties': "uint16",
        'aantalpostransacties': "uint16",
        'aantalfueltransacties': "uint16",
        'depositoyn': "uint8",
        'saldodeposito': "uint32",
        'flexibelsparenyn': "uint8",
        'saldoflexibelsparen': "uint32",
        'kwartaalsparenyn': "uint8",
        'saldokwartaalsparen': "uint32",
        'gemaksbeleggenyn': "uint8",
        'saldogemaksbeleggen': "uint32",
        'participatieyn': "uint8",
        'saldoparticipatie': "uint32",
        'vermogensbeheeryn': "uint8",
        'saldovermogensbeheer': "uint32",
        'saldototaal': "int32",
        'saldolangetermijnsparen': "uint32",
        'aantaltegenrekeningenlaatsteq': "uint16"}

    aggfuncDict = {**aggFuncDictP2, **aggFuncDictP3}
    return aggfuncDict

def getPivotColumns():
            columnsPivot = [
            # "dateeow",
            # "yearweek",
            # "portfolioid",
            # "pakketcategorie",
            # 'overstapserviceyn',
            # 'betaalalertsyn',
            # 'aantalbetaalalertsubscr',
            # 'aantalbetaalalertsontv',
            'roodstandyn',
            'saldoregulatieyn',
            'appyn',
            'aantalloginsapp',
            'aantalloginsweb',
            'activitystatus',
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
            'gemaksbeleggenyn',
            'saldogemaksbeleggen',
            'participatieyn',
            'saldoparticipatie',
            'vermogensbeheeryn',
            'saldovermogensbeheer',
            'saldototaal',
            'saldolangetermijnsparen',
            'aantaltegenrekeningenlaatsteq'
        ]







##


##Pakketcategorie en activitystatus komt er later bij

pass