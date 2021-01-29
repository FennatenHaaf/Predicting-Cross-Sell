""""
File to centrally declare which variables to use and not to use in the analysis in importing, exporting or Analysis.
"""

def getPatColToParseCross(subset="total"):
    columnsToParse1 = [
        "dateeow",
        # "yearweek",
        "portfolioid",
        "pakketcategorie"]

    columnsToParse2 = [
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
    elif subset == "transaction":
        return (columnsToParse1 + columnsToParse3)
    elif subset == "all":
        return (columnsToParse1 + columnsToParse2 + columnsToParse3), (columnsToParse1 + columnsToParse2), \
               (columnsToParse1 + columnsToParse3)
    else:
        print("error getting list")


def getPatColToParseTS(subset="total"):
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

def getPatConvert(subset = "total"):
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

    if subset == "total":
        return (datatypeConvertP1  + datatypeConvertP2 + datatypeConvertP3)
    elif subset == "activity":
        return (datatypeConvertP1 + datatypeConvertP2)
    elif subset == "transaction":
        return (datatypeConvertP1 + datatypeConvertP3)
    elif subset == "all":
        return (datatypeConvertP1  + datatypeConvertP2 + datatypeConvertP3),(datatypeConvertP1 + datatypeConvertP2),\
               (datatypeConvertP1 + datatypeConvertP3)
    else:
        print("error getting dictionary")

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