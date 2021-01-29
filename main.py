from knab_dataprocessor import *
import utils

# Todo Aparte main gecrëerd zodat knabprocessor ook apart te runnen is.
if __name__ == "__main__":

    indirec = "./data"
    outdirec = "./output"
    interdir = "./interdata"
    datatest = data_linking(indirec,interdir,outdirec)

    cross_sec = datatest.create_base_cross_section(date_string="2020-12",
                                                   subsample=True,
                                                   quarterly=True)