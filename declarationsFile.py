""""
File to centrally declare which variables to use and not to use in the 
analysis in importing, exporting or Analysis.

Written for the Quantitative Marketing & Business Analytics seminar
Erasmus School of Economics
"""
import numpy as np


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
        'businessid': 'category',
         'businessType': "category",
          'foundingDate':'uint16',
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
        'dateinstroomweek' : 'datetime64[ns]',
        'instroomjaarweek' : 'datetime64[ns]',
        'instroompakket' : 'category',
        'birthyear': 'uint16',
        'geslacht': 'category',
        'type': 'category',
        'enofyn': 'uint8'
    }

    df_bhk_dict = {
        'boekhoudkoppeling' : 'category'
    }

    time_sets_new = {
        'has_experian_data' : 'uint8',
        'has_business_id': 'uint8',
        'has_account_overlay': 'uint8',
        'indicator_corp_and_retail' : 'uint8',
        'indicator_corp_and_retail_business': 'uint8',
        'iscorporatepersonyn_business': 'uint8',
        'business_id_with_corp_and_retail': 'uint8',
        'retail_id_with_corp_and_retail' : 'uint8'
    }
    time_set_new2 = {
        'converted_period': 'datetime64',
        'indicator_1_inactief': 'uint8',
        'indicator_2_SparenOnlyYN': 'uint8',
        'indicator_3_actief': 'uint8',
        'indicator_4_primaire bank' : 'uint8'
    }

    final_df_dict = {
    'aantal_SBI': 'uint8',
     'aantal_sector':'uint8',
     'aantal_types': 'uint8',
     'aantalatmtransacties_business': 'uint8',
     'aantalatmtransacties_joint': 'uint8',
     'aantalatmtransacties_retail': 'uint16',
     'aantalbetaaltransacties_business': 'uint16',
     'aantalbetaaltransacties_joint': 'uint16',
     'aantalbetaaltransacties_retail': 'uint8',
     'aantalfueltransacties_business': 'uint8',
     'aantalfueltransacties_joint': 'uint8',
     'aantalfueltransacties_retail': 'uint8',
     'aantalloginsapp_business': 'uint16',
     'aantalloginsapp_joint': 'uint16',
     'aantalloginsapp_retail': 'uint16',
     'aantalloginsweb_business': 'uint16',
     'aantalloginsweb_joint': 'uint16',
     'aantalloginsweb_retail': 'uint16',
     'aantalpostransacties_business': 'uint16',
     'aantalpostransacties_joint': 'uint16',
     'aantalpostransacties_retail': 'uint16',
     'aantaltegenrekeningenlaatsteq_business': 'uint16',
     'aantaltegenrekeningenlaatsteq_joint': 'uint16',
     'aantaltegenrekeningenlaatsteq_retail': 'uint16',
     'accountoverlay': 'uint8',
     'activitystatus_business': 'uint8',
     'activitystatus_joint': 'uint8',
     'activitystatus_retail': 'uint8',
     'betalenyn_business': 'uint8',
     'betalenyn_joint': 'uint8',
     'betalenyn_retail': 'uint8',
     'depositoyn_business': 'uint8',
     'depositoyn_joint': 'uint8',
     'depositoyn_retail': 'uint8',
     'flexibelsparenyn_business': 'uint8',
     'flexibelsparenyn_joint': 'uint8',
     'flexibelsparenyn_retail': 'uint8',
     'geslacht_joint': 'category',
     'joint':'uint8',
     'kwartaalsparenyn_business':"uint8",
     'kwartaalsparenyn_joint':"uint8",
     'kwartaalsparenyn_retail':"uint8",
     'retail': "uint8",
     'saldototaal_business': 'int32',
     'saldototaal_joint':'int32',
     'saldototaal_retail': 'int32'
     }

    final_df_dict2 = {'SBIname_on_saldofraction': "category",
     'SBIsectorName_on_saldofraction' : "category",
     'businessType_on_saldofraction': "category",
     'saldototaal_agg': 'int32',
     'saldototaal_fraction': 'float8',
                      'SBIcode_on_saldofraction':"uint8",
                     'SBIsector_on_saldofraction': "category",
                      'period_obs': "datetime64"
    }

    lowercase_input_df_dict = {
        'aantal_sbi':"uint8",
     'businessageinyears': "float16",
     'businesstype': "category",
     'businesstype_on_saldofraction': "category",
     'sbicode': "uint8",
     'sbicode_on_saldofraction': "uint8",
     'sbiname': "category",
     'sbiname_on_saldofraction': "category",
     'sbisector': "category",
     'sbisector_on_saldofraction': "category",
     'sbisectorname': "category",
     'sbisectorname_on_saldofraction': "category"}

    additional_dataprocess_dict = {
        'aantalproducten_totaal': "uint8",
         'aantalproducten_totaal_business': "uint8",
         'aantalproducten_totaal_joint': "uint8",
         'aantalproducten_totaal_retail': "uint8",
         'aantaltransacties_totaal': "uint8",
         'aantaltransacties_totaal_business': "uint16",
         'aantaltransacties_totaal_joint': "uint16",
         'aantaltransacties_totaal_retail': "uint16",
         'logins_totaal': "uint8",
         'logins_totaal_business': "uint8",
         'logins_totaal_joint': "uint8",
         'logins_totaal_retail': "uint8"
    }


    return {**datatypeGeneralActivity, **datatypeActivity, **datatypeTrans, **df_lpp_dict, **df_exp_dict,
            **df_cor_dict, **df_pin_dict, **time_sets_new, **time_set_new2, **final_df_dict, **final_df_dict2,
            **lowercase_input_df_dict, **additional_dataprocess_dict}



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

def getTimeConvertDict():
    """ Define datatypes to import variables as, in order to make it more
    memory efficient"""

    datatypeGeneralActivity = {
        'dateeow': 'count',
        # 'yearweek': "last",
        # 'portfolioid': "nunique",
        'pakketcategorie': "last"}

    datatypeActivity = {
        'overstapserviceyn': "median",
        'betaalalertsyn': "median",
        'aantalbetaalalertsubscr': "sum",
        'aantalbetaalalertsontv': "sum",
        'roodstandyn': "median",
        'saldoregulatieyn': "median",
        'appyn': "median",
        'aantalloginsapp': "sum",
        'aantalloginsweb': "sum",
        'activitystatus': (lambda x: x.mode()[0]  )}

    datatypeTrans = {
        'betalenyn': "median",
        'saldobetalen': "mean",
        'aantalbetaaltransacties': "sum",
        'aantalatmtransacties': "sum",
        'aantalpostransacties': "sum",
        'aantalfueltransacties': "sum",
        'depositoyn': "median",
        'saldodeposito': "mean",
        'flexibelsparenyn': "last",
        'saldoflexibelsparen': "mean",
        'kwartaalsparenyn': "median",
        'saldokwartaalsparen': "mean",
        'gemaksbeleggenyn': "median",
        'saldogemaksbeleggen': "mean",
        'participatieyn': "median",
        'saldoparticipatie': "mean",
        'vermogensbeheeryn': "median",
        'saldovermogensbeheer': "mean",
        'saldototaal': "mean",
        'saldolangetermijnsparen': "mean",
        'aantaltegenrekeningenlaatsteq': "sum"}

    df_lpp_dict = {
        "validfromdate": "first",
        "validfromyearweek": "last",
        # "personid": "category",
        "iscorporatepersonyn": "median",
        'validfromdate_lpp': "first"
    }

    df_exp_dict = {
        'valid_from_dateeow': "first",
        'valid_to_dateeow': "last",
        'age_hh': "median",
        'hh_child': "median",
        'hh_size': "median",
        'income': "median",
        'educat4': "median",
        'housetype': "median",
        'finergy_tp': ( lambda x: x.mode()[0] ),
        'lfase': "median",
        'business': "median",
        'huidigewaarde_klasse': "median"}

    df_cor_dict = {
        'businessid': 'last',
        'businessType': "last",
        'foundingDate': 'last',
        'businessAgeInDays': 'last',
        'businessAgeInMonths': 'last',
        'businessAgeInYears': 'last',
        'foundingYear': 'last',
        'SBIcode': 'last',
        'SBIname': 'last',
        'SBIsector': 'last',
        'SBIsectorName': 'last'
    }

    df_pin_dict = {
        'dateinstroomweek': 'last',
        'instroomjaarweek': 'last',
        'instroompakket': 'last',
        'birthyear': 'last',
        'geslacht': 'last',
        'type': 'last',
        'enofyn': 'last'
    }

    df_bhk_dict = {
        'boekhoudkoppeling' : ( lambda x: x.mode()[0] if x.any() else np.nan )
    }

    time_sets_new = {
        'has_experian_data': 'last',
        'has_business_id': 'last',
        'has_account_overlay': 'last',
        'indicator_corp_and_retail': 'last',
        'indicator_corp_and_retail_business': 'last',
        'iscorporatepersonyn_business': 'last',
        'business_id_with_corp_and_retail': 'last',
        'retail_id_with_corp_and_retail': 'last'
    }

    time_set_new2 = {
        'new_period' : 'median',
        'indicator_1_inactief' : 'median',
        'indicator_2_SparenOnlyYN': 'median',
        'indicator_3_actief': 'median',
        'indicator_4_primaire bank' : 'median'
    }
    return {**datatypeGeneralActivity, **datatypeActivity, **datatypeTrans,
            **df_lpp_dict, **df_exp_dict, **df_cor_dict, **df_pin_dict, **df_bhk_dict, **time_sets_new}

def getColToParseLTSunc():
    ts_list = [
        'dateeow',
     # 'yearweek',
     'portfolioid',
     'pakketcategorie',
     'overstapserviceyn',
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
     'aantaltegenrekeningenlaatsteq',
     # 'valid_from_dateeow',
     # 'valid_to_dateeow',
     'personid',
     'age_hh',
     'hh_child',
     'hh_size',
     'income',
     'educat4',
     'housetype',
     'finergy_tp',
     'lfase',
     'business',
     'huidigewaarde_klasse',
     # 'retail_id_with_corp_and_retail',
     'iscorporatepersonyn',
     'validfromdate_lpp',
     # 'indicator_corp_and_retail',
     'businessid',
     'businessType',
     'businessAgeInDays',
     'foundingYear',
     'SBIcode',
     'SBIname',
     'SBIsector',
     'SBIsectorName',
     # 'indicator_corp_and_retail_business',
     # 'iscorporatepersonyn_business',
     # 'business_id_with_corp_and_retail',
     'boekhoudkoppeling',
     'has_account_overlay',
     'has_business_id',
     'has_experian_data'
    ]
    return ts_list

def getPersonAggregateDict():
    """ Define datatypes to import variables as, in order to make it more
    memory efficient"""

    datatypeGeneralActivity = {
        # 'dateeow': 'last',
        # 'yearweek': "last",
        'portfolioid': "nunique",
        'pakketcategorie': (lambda x: x.mode()[0], 'nunique') }

    datatypeActivity = {
        'overstapserviceyn': "median",
        'betaalalertsyn': "median",
        'aantalbetaalalertsubscr': "sum",
        'aantalbetaalalertsontv': "sum",
        'roodstandyn': "median",
        'saldoregulatieyn': "median",
        'appyn': "median",
        'aantalloginsapp': "sum",
        'aantalloginsweb': "sum",
        'activitystatus': (lambda x: x.mode()[0] , 'nunique')}

    datatypeTrans = {
        'betalenyn': "median",
        'saldobetalen': "max",
        'aantalbetaaltransacties': "max",
        'aantalatmtransacties': "max",
        'aantalpostransacties': "max",
        'aantalfueltransacties': "max",
        'depositoyn': "max",
        'saldodeposito': "max",
        'flexibelsparenyn': "max",
        'saldoflexibelsparen': "max",
        'kwartaalsparenyn': "max",
        'saldokwartaalsparen': "max",
        'gemaksbeleggenyn': "max",
        'saldogemaksbeleggen': "max",
        'participatieyn': "max",
        'saldoparticipatie': "max",
        'vermogensbeheeryn': "max",
        'saldovermogensbeheer': "max",
        'saldototaal': ("max","mean"),
        'saldolangetermijnsparen': "mean",
        'aantaltegenrekeningenlaatsteq': "sum"}

    df_lpp_dict = {
        "validfromdate": "first",
        "validfromyearweek": "last",
        # "personid": "category",
        "iscorporatepersonyn": "median",
        'validfromdate_lpp': "first"
    }

    df_exp_dict = {
        'valid_from_dateeow': "first",
        'valid_to_dateeow': "last",
        'age_hh': "median",
        'hh_child': "median",
        'hh_size': "median",
        'income': "median",
        'educat4': "median",
        'housetype': "median",
        'finergy_tp': ( lambda x: x.mode()[0] ),
        'lfase': "median",
        'business': "median",
        'huidigewaarde_klasse': "median"}

    df_cor_dict = {
        'businessid': 'last',
        'businessType': "last",
        'foundingDate': 'last',
        'businessAgeInDays': 'last',
        'businessAgeInMonths': 'last',
        'businessAgeInYears': 'last',
        'foundingYear': 'last',
        'SBIcode': 'last',
        'SBIname': 'last',
        'SBIsector': 'last',
        'SBIsectorName': 'last'
    }

    df_pin_dict = {
        'dateinstroomweek': 'min',
        'instroomjaarweek': 'last',
        'instroompakket': 'last',
        'birthyear': 'last',
        'geslacht': 'last',
        'type': 'last',
        'enofyn': 'last'
    }

    df_bhk_dict = {
        'boekhoudkoppeling' : (( lambda x: x.mode()[0] if x.any() else np.nan ) , 'nunique')
    }

    time_sets_new = {
        'has_experian_data': 'max',
        'has_business_id': 'max',
        'has_account_overlay': 'max',
        # 'indicator_corp_and_retail': 'last',
        # 'indicator_corp_and_retail_business': 'last',
        # 'iscorporatepersonyn_business': 'last',
        # 'business_id_with_corp_and_retail': 'last',
        # 'retail_id_with_corp_and_retail': 'last'
    }

    last_set = {
        'has_new_val'
    }

    time_set_new2 = {
        # 'converted_period' : 'max',
        'indicator_1_inactief' : ['max','sum'],
        'indicator_2_SparenOnlyYN': ['max','sum'],
        'indicator_3_actief': ['max','sum'],
        'indicator_4_primaire bank' : ['max','sum']
    }

    return {**datatypeGeneralActivity, **datatypeActivity, **datatypeTrans,
            **df_lpp_dict, **df_exp_dict, **df_cor_dict, **df_pin_dict, **df_bhk_dict, **time_sets_new, **time_sets_new}

 ###-----------CROSS_SECTION ---------------------------####



def get_cross_section_agg( list_to_get ):
    count_list = [

     'aantalatmtransacties_business' ,
     'aantalatmtransacties_joint',
     'aantalatmtransacties_retail',
     'aantalbetaaltransacties_business',
     'aantalbetaaltransacties_joint',
     'aantalbetaaltransacties_retail',
     'aantalfueltransacties_business',
     'aantalfueltransacties_joint',
     'aantalfueltransacties_retail',
     'aantalloginsapp_business',
     'aantalloginsapp_joint',
     'aantalloginsapp_retail',
     'aantalloginsweb_business',
     'aantalloginsweb_joint' ,
     'aantalloginsweb_retail' ,
     'aantalpostransacties_business',
     'aantalpostransacties_joint',
     'aantalpostransacties_retail',
     'aantaltegenrekeningenlaatsteq_business',
     'aantaltegenrekeningenlaatsteq_joint',
     'aantaltegenrekeningenlaatsteq_retail',
        'aantaltransacties_totaal',
        'aantaltransacties_totaal_business',
        'aantaltransacties_totaal_joint',
        'aantaltransacties_totaal_retail',
        'logins_totaal',
        'logins_totaal_business',
        'logins_totaal_joint',
        'logins_totaal_retail',
        'log_aantaltransacties_totaal',
        'log_logins_totaal',

    ]

    balance_at_moment = [
    'saldototaal',
    'saldototaal_business',
    'saldototaal_joint',
    'saldototaal_retail',
        'log_saldototaal',
         ]

    moment_counts = [
    'aantal_sbi' ,
    'aantal_sector' ,
    'aantal_types'
        ]

    categorical = [
        'accountoverlay'
        'activitystatus_business'
        'activitystatus_joint',
        'activitystatus_retail',
        'age_hh',
         'businesstype',
         'businesstype_on_saldofraction'
         'birthyear',
        'business',
        'educat4',
        'finergy_tp',
        'geslacht',
        'geslacht_joint',
        'sbicode',
        'sbicode_on_saldofraction',
        'sbiname',
        'sbiname_on_saldofraction',
        'sbisector',
        'sbisector_on_saldofraction',
        'sbisectorname',
        'sbisectorname_on_saldofraction',
        'lfase',
        'hh_child',
        'hh_size',
        'housetype',
        'huidigewaarde_klasse',
        'income',
        'activitystatus',
        'activitystatus_business',
        'activitystatus_joint',
        'age',
        'age_bins',
        'birthyear',
        'businessageinyears_bins',
        'businesstype_on_saldofraction',
         ]

    indicators = ['betalenyn_business',
     'betalenyn_joint',
     'betalenyn_retail',
     'depositoyn_business',
     'depositoyn_joint',
     'depositoyn_retail',
     'flexibelsparenyn_business',
     'flexibelsparenyn_joint',
     'flexibelsparenyn_retail',
     'kwartaalsparenyn_business',
     'kwartaalsparenyn_joint',
     'kwartaalsparenyn_retail',
     ]

    dependent_variables = [
        'retail_max',
        'retail_prtf_counts'
        'portfolio_total_counts',
        'has_bus_prtf',
        'has_bus_ret_prtf',
        'has_jnt_prtf',
        'has_jnt_ret_prtf',
        'has_ret_prtf',
        'joint_max',
        'joint_prtf_counts',
        'business_max',
        'business_prtf_counts',
        'accountoverlay',
        'accountoverlay_dummy',
        'accountoverlay_max',
        'aantalproducten_totaal',
        'aantalproducten_totaal_business',
        'aantalproducten_totaal_joint',
        'aantalproducten_totaal_retail',
    ]

    remainder_dict = {
                               'joint': 'max',
               'retail': 'max',
               'businessageinyears': 'first',
                'saldototaal_fraction': 'last'
                }

    if list_to_get == "count_list":
        return count_list
    elif list_to_get == "balance_at_moment":
        return balance_at_moment
    elif list_to_get == "moment_counts":
        return moment_counts
    elif list_to_get == "categorical":
        return categorical
    elif list_to_get == "indicators":
        return indicators
    elif list_to_get == "remainder_dict":
        return remainder_dict
    elif list_to_get == "dependent_variables":
        return dependent_variables
    elif list_to_get == 'all':
        return count_list + balance_at_moment + moment_counts + categorical + indicators + dependent_variables + list(set(
            remainder_dict))
    else:
        print("Invalid value for list_to_get")

