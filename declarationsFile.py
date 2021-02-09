""""
File to centrally declare which variables to use and not to use in the 
analysis in importing, exporting or Analysis.

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""

def getPatColToParseCross(subset="total"):
    """ Method to select which specific columns are imported from the Knab
    transaction and activity datasets"""
    
    columnsToParse1 = [
        "dateeow",
        # "yearweek",
        "portfolioid",
        "pakketcategorie"]

    columnsToParse2 = [ # activity
        'overstapserviceyn',
        'betaalalertsyn',
        # 'aantalbetaalalertsubscr',
        # 'aantalbetaalalertsontv',
        'roodstandyn',
        'saldoregulatieyn',
        'appyn',
        'aantalloginsapp',
        'aantalloginsweb',
        'activitystatus'
    ]

    columnsToParse3 = [  # transactions
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
        # De onderstaande variabelen zijn in sommige jaren altjd 0 dus
        # deze gebruiken we niet!
        # 'gemaksbeleggenyn',  
        # 'saldogemaksbeleggen',
        # 'participatieyn',
        # 'saldoparticipatie',
        # 'vermogensbeheeryn',
        # 'saldovermogensbeheer',
        'saldototaal',
        'saldolangetermijnsparen',
        'aantaltegenrekeningenlaatsteq'
    ]
    if subset == "total":
        return (columnsToParse1 + columnsToParse2 + columnsToParse3)
    elif subset == "activity":
        return (columnsToParse1 + columnsToParse2)
    elif subset == "transactions":
        return (columnsToParse1 + columnsToParse3)
    elif subset == "all":
        return (columnsToParse1 + columnsToParse2 + columnsToParse3), (columnsToParse1 + columnsToParse2), \
               (columnsToParse1 + columnsToParse3)
    else:
        print("error getting list")


def getConvertDict():
    """ Define datatypes to import variables as, in order to make it more
    memory efficient"""
    
    datatypeGeneralActivity = {
        'dateeow': "datetime64",
        'yearweek': "string",
        'portfolioid': "category",
        'pakketcategorie': "category"}

    datatypeActivity = {
        'overstapserviceyn': "uint8",
        'betaalalertsyn': "uint8",
        'aantalbetaalalertsubscr': "uint16",
        'aantalbetaalalertsontv': "uint16",
        'roodstandyn': "uint8",
        'saldoregulatieyn': "uint8",
        'appyn': "uint8",
        'aantalloginsapp': "uint16",
        'aantalloginsweb': "uint16",
        'activitystatus': "category"}

    datatypeTrans = {
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

    df_lpp_dict = {
        "validfromdate" : "datetime64",
        "validfromyearweek" : "int32",
        "personid": "category",
        "iscorporatepersonyn": "uint8"
    }

    df_exp_dict = {
        'valid_from_dateeow':"datetime64",
        'valid_to_dateeow':"datetime64",
        'age_hh' :"uint8",
        'hh_child' :"uint8",
        'hh_size' :"uint8",
        'income':"uint8" ,
        'educat4':"uint8",
        'housetype':"uint8",
        'finergy_tp': "category",
        'lfase':"uint8",
        'business':"uint8",
        'huidigewaarde_klasse':"uint8" }

    df_cor_dict = {
         'personid':"category",
         'businessType': "category",
          'foundingDate':'datetime64',
         'businessAgeInDays':'uint16',
        'businessAgeInMonths': 'uint16',
        'businessAgeInYears': 'float16',
         'foundingYear':'uint16',
         'SBIcode':'category',
         'SBIname':'category',
         'SBIsector':'category',
         'SBIsectorName':'category'
    }

    df_pin_dict = {
        'dateinstroomweek' : 'datetime64',
        'instroomjaarweek' : 'datetime64',
        'instroompakket' : 'category',
        'birthyear': 'uint16',
        'geslacht': 'category',
        'type': 'category',
        'enofyn': 'uint8'
    }

    return {**datatypeGeneralActivity, **datatypeActivity, **datatypeTrans, 
            **df_lpp_dict, **df_exp_dict, **df_cor_dict}



###----TIME SERIES ---------------------------------------------------------

def getPatColToParseTS(subset="total"):
    columnsToParse1 = [
        "dateeow",
        # "yearweek",
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

    if subset == "total":
        return (columnsToParse1 + columnsToParse2 + columnsToParse3)
    elif subset == "activity":
        return (columnsToParse1 + columnsToParse2)
    elif subset == "transaction":
        return (columnsToParse1 + columnsToParse3)
    elif subset == "all":
        return (columnsToParse1 + columnsToParse2 + columnsToParse3), (columnsToParse1 + columnsToParse2), \
               (columnsToParse1 + columnsToParse3)
    else:
        print("error getting list")

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
    return columnsPivot

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