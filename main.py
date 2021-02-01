import knab_dataprocessor as KD
import utils

# Todo Aparte main gecrëerd zodat knabprocessor ook apart te runnen is.
if __name__ == "__main__":

    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    test = KD.dataProcessor(indirec,interdir,outdirec)
    finalDF = test.KD.link_data()

    cross_sec = datatest.create_base_cross_section(date_string="2020-12",
                                                   subsample=True,
                                                   quarterly=True)