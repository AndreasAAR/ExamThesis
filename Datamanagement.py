import numpy as np, numpy
import pandas as pd, pandas
import matplotlib
import collections
import StatMethods as sm
import statistics as stat
import Graphs as grap


###Clean Table extraction##

def getCleanData():
    data = np.loadtxt("dna_amplification.txt")
    data2 = []
    data2.append(data)
    data = data2
    #Adding cancer types
    table = pd.read_csv('dna_amplification.txt', header = None, sep = " ")
    rowNames = pd.read_csv("neoplasm_types.txt", header = None, sep ="\t")
    rowNames = rowNames[1]
    table["Type"] = rowNames
    #Removing NA collumn
    table = table.drop([table.columns[393]] ,  axis='columns')
    return table

def equalizeClass(lowestRep):
    print("IM METHOD")

data = getCleanData()



