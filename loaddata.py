import numpy as np
import pandas as pd


def loadtxtAndcsv_data(fileName, split, dataType):  # load
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)

def loaddata():
    data1 = loadtxtAndcsv_data(r'C:\Users\dell\Desktop\Git\datatrain.csv', ",", np.float64)
    data2 = loadtxtAndcsv_data(r'C:\Users\dell\Desktop\Git\datatest.csv', ",", np.float64)
    data_train = data1[:, 0:3]
    targets_train = data1[:, -3]
    data_test = data2[:, 0:3]
    targets_test = data2[:, -3]
    return data_train,data_test,targets_train,targets_test
