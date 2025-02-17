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
import numpy as np
import matplotlib.cm as cm
import copy


# =============================================================================
# PRINTING INTERMEDIATE RESULTS
# =============================================================================

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
        
        
# =============================================================================
# METHODS FOR PLOTTING & VISUALIZING
# =============================================================================
        
def visualize_matrix(self,matrix,x_axis,y_axis,xlabel,ylabel,title,
                         diverging = False, annotate = True, xticks = None,
                         yticks = None):
        """Visualize a 2D matrix in a figure with labels and title"""
        if not self.visualize_data:
            return

        plt.rcParams["axes.grid"] = False
        fig, ax = plt.subplots(figsize=(14, 8))

        # Define the colors
        if diverging:
            colMap = copy.copy(cm.get_cmap("coolwarm"))      
        else:
            colMap = copy.copy(cm.get_cmap("viridis"))
            colMap.set_under(color='white') # set under deals with values below minimum
            # colMap.set_bad(color='black') # set bad deals with color of nan values
            
        # Now plot the values
        im = ax.imshow(matrix, cmap = colMap)
        # set the max color to >1  so that the lightest areas are not too light
        # below the min we want it to be white!
        if diverging:
            im.set_clim(-1.5, 1.5)  
        else:
            im.set_clim(1e-15, 1.2)  
           
        #Set font sizes
        labelfont = 20
        tickfont = 19
        annotfont = 30
        titlefont = 20
        
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(x_axis)))
        ax.set_yticks(np.arange(len(y_axis)))
        # ... and label them with the respective list entries
        if not( isinstance(xticks, type(None)) ): 
            ax.set_xticklabels(xticks,fontsize = tickfont)
        else: 
            ax.set_xticklabels(x_axis,fontsize = tickfont)
        if not( isinstance(yticks, type(None)) ): 
            ax.set_yticklabels(yticks,fontsize = tickfont)
        else:
            ax.set_yticklabels(y_axis,fontsize = tickfont)
        
        #ax.xaxis.set_label_position('top') 
        #ax.xaxis.tick_top()
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        
        #Label the axes   
        ax.set_xlabel(xlabel,fontsize = labelfont)
        ax.set_ylabel(ylabel,fontsize = labelfont)

        # Loop over data dimensions and create text annotations.
        if annotate:
            for i in range(len(y_axis)):
                for j in range(len(x_axis)):
                    text = ax.text(j, i, round(matrix[i, j],3),
                                   ha="center", va="center", color="w",
                                   fontsize = annotfont)
        
        ax.set_title(title,fontsize = titlefont, fontweight='bold')
        fig.tight_layout()
        
        #cbar = plt.colorbar()
        #cbar.set_label('Probability')
        plt.show()
      
        
def plotFinergyCounts(df_experian, ids):
    """Makes a plot of how frequently each finergy type occurs in a dataset"""
    
    sns.set(rc={'figure.figsize':(20,10)})
    sns.set_style("whitegrid")

    df_experian = df_experian[df_experian["personid"].isin(ids)] 
    print("plotting finergy counts")
    graph =sns.countplot(x="finergy_tp", data = df_experian)
    
    # set labels
    graph.set_xlabel("Finergy type",fontsize = 21)
    graph.set_ylabel("Count",fontsize = 21)
    graph.tick_params(axis="x", labelsize=18) 
    graph.tick_params(axis="y", labelsize=20) 
    plt.show()
    
        
def plotCategorical(df, name, xlabel = None, annotate = True,
                    colours = None, onecol= False):
    """Make a plot of the frequency of each category for a categorical variable"""
    
    sns.set(font_scale=2,rc={'figure.figsize':(12,8)})
    sns.set_style("whitegrid")
    if onecol:
        graph = sns.countplot(x=name, data = df, color=colours)
    else:
        graph = sns.countplot(x=name, data = df, palette=colours)
    
    
    graph.set_xticklabels(graph.get_xticklabels(),rotation=30,
                          horizontalalignment='right', fontweight='light',
                          fontsize=22)
    
    if annotate:
        ## We want to label the bars with the height, to see what the exact counts are
        for p in graph.patches:
            value = p.get_height()
            # locations of where to put the labels
            x = p.get_x() + (p.get_width() / 2) - 0.2 
            y = p.get_y() + p.get_height()
            graph.annotate(round(value,0), (x, y), size = 22)
    
    #Set labels
    graph.set_xlabel(xlabel,fontsize = 22)
    
    plt.show()
    
    
def plotCoocc(df, names, annotate = True, colors = "plasma", cent = 0,
              make_perc=True, make_total_perc=False): 
    """Plot cooccurrences"""
    
    sns.set(font_scale=3,rc={'figure.figsize':(15,12)})
    
    coocc= df[names].T.dot(df[names])
    graph = sns.heatmap(coocc, center=cent, annot = annotate, 
                        cmap=colors, fmt='g',cbar=False) 
    
    
    labels = ["business","retail","joint","accountoverlay"]  
        
    graph.set_yticklabels(labels,rotation=0,
                          horizontalalignment='right', fontweight='light',
                          fontsize=30)
    graph.set_xticklabels(labels,rotation=0,
                          horizontalalignment='center', fontweight='light',
                          fontsize=30)
    plt.show()
    
    if make_perc:
        # Now also make the percentages version
        coocc_diagonal = np.diagonal(coocc)
        with np.errstate(divide='ignore', invalid='ignore'):
            coocc_perc =  np.nan_to_num(round(np.true_divide(coocc,
                                                     coocc_diagonal[:, None]),3))
        
        graph = sns.heatmap(coocc_perc, center=cent, annot = annotate, 
                            cmap=colors, fmt='g',cbar=False,vmin=0.2, vmax=1.2) 
    
        graph.set_yticklabels(labels,rotation=0,
                              horizontalalignment='right', fontweight='light',
                              fontsize=30)
        graph.set_xticklabels(labels,rotation=0,
                              horizontalalignment='center', fontweight='light',
                              fontsize=30)
        
        plt.show()
    
    # plot cooccurrences, but with percentage of total labelled in brackets
    if make_total_perc: 
        
        # create an 'annotate' dataframe which is used as labels in the heatmap
        annotate = coocc.copy()
        for i in range(0,annotate.shape[0]):
            for j in range(0,annotate.shape[1]):
                annotate.iloc[i,j] = f"{coocc.iloc[i,j]} ({round((coocc.iloc[i,j]/len(df))*100,2)}%)"
        
        # make the plot
        sns.set(font_scale=3,rc={'figure.figsize':(16,13)})
        graph = sns.heatmap(coocc, center=cent, annot = annotate, 
                            cmap=colors, fmt='', 
                            cbar=False) 
        graph.set_yticklabels(labels,rotation=0,
                              horizontalalignment='right', fontweight='light',
                              fontsize=30)
        graph.set_xticklabels(labels,rotation=0,
                              horizontalalignment='center', fontweight='light',
                              fontsize=30)
        
        plt.show()
    
    
def plot_portfolio_changes(dataset, varvector, percent = True,
                           ignore_0 = True, legend = True, xlabel = None,
                           plot=True, labelwidth = 30,
                           colours = "coolwarm"):
  """Make a plot of cross-sell occurrences or changes in portfolio ownerships
  between time periods in a full dataset"""
  
  # Select the different values for portfolio change occurring in the 
  # dataset
  column_values = dataset[varvector].values.ravel()
  unique_values = pd.Series(pd.unique(column_values)).sort_values().dropna()

  # create a dataframe that contains for each portfolio type how often each
  # change value occurs
  df = pd.DataFrame(columns = varvector, index = unique_values)
  for var in varvector:
      group = dataset[var]
      for value in unique_values: 
        if ((value == 0) & (ignore_0 == True)):
            True # We do nothing
        else: 
            count = int((group == value).sum())
            perc = (count/len(group))*100
            if (percent):
              df.loc[value,var] = perc
            else:       
              df.loc[value,var] = count
              
  if plot:# Now make the plot
    sns.set(font_scale=1.1, rc={'figure.figsize':(20,6)})
    sns.set_style("whitegrid")

    # stack so we get a 'long form' df for plotting
    dfstacked = df.stack().reset_index() 

    fig, graph = plt.subplots()
    sns.barplot(x = dfstacked["level_1"], y =dfstacked[0],
                data = dfstacked, hue = dfstacked["level_0"],
                palette=colours, ax=graph)
    
    # We want to label the bars with the height, to see what the exact counts are
    for p in graph.patches:
        value = p.get_height()
        
        # locations of where to put the labels
        x = p.get_x() + p.get_width() / 2 - 0.05 
        
        if ((p.get_height() <0.1) & (p.get_height()>0.015)):
            y = p.get_y() + p.get_height() + 0.015
        elif ((p.get_height() <0.01)):
             y = p.get_y() + p.get_height() - 0.005
        else:
            y = p.get_y() + p.get_height() 

        if percent:
          if (value !=0):
              graph.annotate(f"{round(value,2)}%", (x, y), size = 18)
        else:
          if (value !=0):
              graph.annotate(int(value), (x, y), size = 20)

    # Make labels for the axes
    graph.set_xlabel(xlabel,fontsize = 20)
    if percent:
        graph.set_ylabel(None,fontsize = 20)
    else:
        graph.set_ylabel("Number of portfolio ownership changes over all periods",fontsize = 20)
        
    # Add legend
    if legend:
        plt.legend(title = None, loc="lower left",bbox_to_anchor=(0.1, -0.3), ncol = len(graph.lines),
                   fontsize = 20)
    else:
        plt.legend([],[], frameon=False)
    
    # Add the ticks to the axes
    labels = ["business","retail","joint","accountoverlay"]  
    graph.set_xticklabels(labels,rotation=0,
                          horizontalalignment='center', fontweight='light',
                          fontsize=20)
    graph.tick_params(axis="y", labelsize=17)
    plt.show()

  return df



def plot_portfolio_changes_stacked(dataset, varvector, percent = True,
                           ignore_0 = True, legend = True, xlabel = None,
                           plot=True, labelwidth = 30,
                           colours = "coolwarm"):
  """Plots portfolio changes in a horizontal stacked bar chart"""
    
  # Select the different values for portfolio change occurring in the 
  # dataset
  column_values = dataset[varvector].values.ravel()
  unique_values = pd.Series(pd.unique(column_values)).sort_values().dropna()
  if (ignore_0):
      unique_values.pop(0) # remove the option 0

  # create a dataframe that contains for each portfolio type how often each
  # change value occurs
  df = pd.DataFrame(columns = varvector, index = unique_values)
  for var in varvector:
      group = dataset[var]
      for value in unique_values: 
        count = int((group == value).sum())
        perc = (count/len(group))*100
        if (percent):
          df.loc[value,var] = perc
        else:       
          df.loc[value,var] = count
  df= df.set_index(unique_values)
  df= df.transpose()

  # NOW MAKE THE PLOT
  if plot:
    sns.set(font_scale=1.1, rc={'figure.figsize':(17,7)})
    sns.set_style("whitegrid")

    graph=df.plot.barh(stacked=True ,cmap = colours)
    
    labels = ["business","retail","joint","accountoverlay"]  
    graph.set_yticklabels(labels,rotation=0,
                          horizontalalignment='right', fontweight='light',
                          fontsize=21)
    
    graph.set_xlabel("Percentage",fontsize = 21)
    graph.set_ylabel(None,fontsize = 21)
    graph.tick_params(axis="x", labelsize=20)
  
    graph.legend(title = None, loc="upper left",bbox_to_anchor=(0.07, 1.14), ncol = len(unique_values),
                   fontsize = 18)
    plt.show()
   
  return df



def plotEvaluationMetrics(dfacc,dfsens,var,colours = ["#62aede","#de9262"],
                          dash = "solid"):
    """Plot accuracy and sensitivity for different thresholds in one graph 
    for one portfolio type"""
    
    # Get only the accuracy 
    data = pd.concat([dfacc.rename(columns={var: "accuracy"})["accuracy"].reset_index(drop=True),
                        dfsens.rename(columns={var: "sensitivity"})["sensitivity"].reset_index(drop=True), 
                        dfacc["threshold_high"].reset_index(drop=True)],axis=1)
    
    # Stack the data so it is in 'long form'(necessary for plotting)
    data= data.set_index("threshold_high").stack().reset_index()
    
    # make the plot
    sns.lineplot(x=data["threshold_high"], y = data[0],
                 hue=data["level_1"],
                 data=data,
                 palette=colours,
                 linestyle = dash,
                 )



# =============================================================================
# OTHER EXPLORE METHODS
# =============================================================================
    
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


