# -*- coding: utf-8 -*-
"""
This file contains extra functions for gaining some insight into the Knab
data 

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

import pandas as pd
import utils
import seaborn as sns
import matplotlib.pyplot as plt

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
    
def plotFinergyCounts(df_experian, ids):
    sns.set(rc={'figure.figsize':(15,10)})
    df_experian = df_experian[df_experian["personid"].isin(ids)] 
    print("plotting finergy counts")
    graph =sns.countplot(x="finergy_tp", data = df_experian)
    plt.show()
    

def mostCommonDict(data, nameList, numberOfValues=1):
    dictOfTables = {}
    for columnName in nameList:
        dictOfTables[columnName] = mostCommon(data, columnName, numberOfValues, True)

def recurringValues(data, indexCol, colToTest, threshold = 1):
    def filterMultiple(x):
        return x.count() > threshold
    return (data.groupby(indexCol)[colToTest].filter(filterMultiple).count())

def uniqueValsList(data, col_list):
    for item in col_list:
        print(f"Unique values in column {item} :",data[item].unique().shape[0])

def checkV1(data, value, column="personid"):
    return data[data[column] == value]

def checkAVD(data, column):
    for value in data[column].unique():
        yield data[data[column] == value]

def checkAVL(list):
    for value in list:
        yield value
        
        
def plotCategorical(df, name, annotate = True):
    sns.set(font_scale=2,rc={'figure.figsize':(15,10)})
    graph = sns.countplot(x=name, data = df)
    
    
    graph.set_xticklabels(graph.get_xticklabels(),rotation=30,
                          horizontalalignment='right', fontweight='light',
                          fontsize=18)
    
    if annotate:
        ## We want to label the bars with the height, to see what the exact counts are
        for p in graph.patches:
            value = p.get_height()
            # locations of where to put the labels
            x = p.get_x() + p.get_width() / 2 - 0.05 
            y = p.get_y() + p.get_height()
            graph.annotate(value, (x, y), size = 16)
    
    
    plt.show()
    
        
    
    
### DATA EXPLORATION METHODS
#TODO show PA explorer in here
def exploreSets(self, link_corp = False, link_pat = False):
    df_exp = pd.read_csv(f"{self.indir}/experian.csv")
    df_lpp= pd.read_csv(f"{self.indir}/linkpersonportfolio.csv")
    df_bhk = pd.read_csv(f"{self.indir}/portfolio_boekhoudkoppeling.csv")
    df_pin = pd.read_csv(f"{self.indir}/portfolio_info.csv")
    df_pst = pd.read_csv(f"{self.indir}/portfolio_status.csv")
    df_bhk.groupby("personid")["portfolioid"].count()
    
    #person id's per protfolio and vice versa
    nOfPIDperPort = df_lpp.groupby("portfolioid")["personid"].count().sort_values(ascending = False)
    nOfPortPerPID = df_lpp.groupby("personid")["portfolioid"].count().sort_values(
        ascending=False)
    
    col_list = ["personid", "portfolioid", "accountoverlayid", "accountid"]
    uniqueValsList(df_bhk, col_list)
    recurringValues(df_bhk, "portfolioid", "personid", threshold = 1)
    
    lpp_col_to_test = ["personid"]
    uniqueValsList(df_lpp, lpp_col_to_test)
    
    ##EXP
    (df_exp.groupby("personid")["valid_from_dateeow"].count() > 1).sum()
    
    #Link bhk en lpp
    df_bhk_lpp = pd.merge(df_bhk, df_lpp[["personid", "portfolioid", "iscorporatepersonyn"]], on=["personid", "portfolioid"])
    print("Number of corporate portfolios for overlay in lpp :", df_bhk_lpp["iscorporatepersonyn"].sum())
    
    #LPP
    df_lpp = utils.doConvertFromDict(df_lpp)
    df_lpp_table_port_corp_retail = df_lpp.groupby("portfolioid")["iscorporatepersonyn"].mean()
    df_lpp_table_port_corp_retail = df_lpp_table_port_corp_retail.reset_index()
    df_lpp_table_port_corp_retail.loc[:,"indicator_corp_and_retail"] = 0
    lpp_index_both = pd.eval(" (df_lpp_table_port_corp_retail['iscorporatepersonyn'] > 0) & "
                             "(df_lpp_table_port_corp_retail['iscorporatepersonyn'] < 1) ")
    df_lpp_table_port_corp_retail.loc[lpp_index_both,"indicator_corp_and_retail"] = 1
    df_lpp_table_port_corp_retail.drop("iscorporatepersonyn", axis=1, inplace=True)
    df_lpp.drop(["validfromdate", "validfromyearweek", "validtodate"], axis=1, inplace=True)
    df_lpp = pd.merge(df_lpp,df_lpp_table_port_corp_retail, on = "portfolioid")
    
    index_link_cr = pd.eval("df_lpp['indicator_corp_and_retail'] == 1")
    index_corp_pers = pd.eval("df_lpp['iscorporatepersonyn'] == 1")
    portid_link_cr = df_lpp.loc[index_link_cr, "portfolioid"].unique()
    persid_link_cr = df_lpp.loc[index_link_cr, "personid"].unique()
    persid_no_link_cr = df_lpp.loc[~index_link_cr, "personid"].unique()
    id_link_cr_corp = df_lpp.loc[(index_link_cr & index_corp_pers) , "personid"].unique()
    id_link_cr_ret = df_lpp.loc[(index_link_cr & ~index_corp_pers), "personid"].unique()
    id_no_link_cr_corp = df_lpp.loc[(~index_link_cr & index_corp_pers), "personid"].unique()
    id_no_link_cr_ret = df_lpp.loc[(~index_link_cr & ~index_corp_pers), "personid"].unique()
    
    df_lpp_linked = df_lpp[pd.eval("df_lpp['personid'].isin(persid_link_cr)")]
    ##LPP en EXP
    pd.DataFrame(df_exp["personid"].unique()).isin(df_lpp_linked.loc[df_lpp_linked["iscorporatepersonyn"] == 0, "personid"])
    exp_lpp_linked = pd.merge(df_exp, df_lpp_linked[["personid", "portfolioid"]], on=["personid"])
    exp_lpp_linked.sort_values(["valid_to_dateeow","personid","portfolioid"], inplace= True)
    exp_lpp_linked["portfolioid"].unique()
    
    exp_lpp_groupby = exp_lpp_linked.groupby("personid")["valid_from_dateeow"].count()
    print( exp_lpp_groupby.index.isin(exp_lpp_groupby[exp_lpp_groupby > 1].index).sum() )
    temp_list = exp_lpp_groupby.index.isin(exp_lpp_groupby[exp_lpp_groupby > 1].index)
    exp_lpp_groupby.reset_index()["personid"][temp_list]
    pd.Series(exp_lpp_groupby.reset_index()["personid"][temp_list])
    
    ##BHK
    df_bhk["personid"].isin(persid_link_cr).sum()
    df_bhk["personid"].isin(id_link_cr_corp).sum()
    df_bhk["personid"].isin(id_no_link_cr_corp).sum()
    df_bhk["personid"].isin(id_link_cr_ret).sum()
    df_bhk["personid"].isin(id_no_link_cr_ret).sum()
    id_link_cr_ret.isin(id_no_link_cr_ret)
    
    n1 = df_bhk.groupby("personid")[["personid","portfolioid", "boekhoudkoppeling", "valid_from_dateeow", "valid_to_dateeow"]]. \
        filter(lambda x: x["boekhoudkoppeling"].unique().shape[0] > 1)
    n1.drop_duplicates(inplace=True)
    n2 = df_bhk.groupby("personid")[["personid", "portfolioid","boekhoudkoppeling", "valid_from_dateeow", "valid_to_dateeow"]]. \
        filter(lambda x: x["boekhoudkoppeling"].unique().shape[0] == 1)
    n2.drop_duplicates(inplace=True)
    
    n3 = df_bhk.groupby("portfolioid")[["personid","portfolioid", "boekhoudkoppeling", "valid_from_dateeow", "valid_to_dateeow"]]. \
        filter(lambda x: x["boekhoudkoppeling"].unique().shape[0] > 1)
    n3.drop_duplicates(inplace=True)
    n4 = df_bhk.groupby("portfolioid")[["personid", "portfolioid","boekhoudkoppeling", "valid_from_dateeow", "valid_to_dateeow"]]. \
        filter(lambda x: x["boekhoudkoppeling"].unique().shape[0] == 1)
    n4.drop_duplicates(inplace=True)

    # bhk_to_corp_pivot = pd.pivot_table(df_bhk_lpp,vales = "","index = "boekhoudkoppeling", columns =  "iscorporatepersonyn", aggfunc= "count")
    #Portfolio Status
    df_pst[df_pst["outflow_date"] == "9999-12-31"] = self.endDate
    df_pst["outflow_date"] = pd.to_datetime(df_pst["outflow_date"])
    
    if link_corp:
        if self.df_corporate_details.empty:
            self.processCorporateData()
        df_bhk[df_bhk["personid"].isin(self.df_corporate_details["personid"].unique())]
    
    if link_pat:
        columns_to_use = ["portfolioid"]
        if self.df_pat.empty:
            readArgs = {"usecols":columns_to_use}
            self.importPortfolioActivity()



   
def explorePA(self):
    pat = self.df_pat
    
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
    
    patSub1 = self.df_pat.loc[:, patSubID1].copy()
    patSub2 = self.df_pat.loc[:, patSubID2].copy()
    patSub3 = self.df_pat.loc[:, patSubID3].copy()
    
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


