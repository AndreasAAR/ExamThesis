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

def barrClasses(lowestRep, dataframe):
    dataframe = dataframe.groupby('Type').filter(lambda x : len(x) > lowestRep)
    return dataframe

def equalizeClasses(lowestRep, dataframe):
    dataframe = barrClasses(lowestRep,dataframe)
    min = dataframe['Type'].value_counts().min()
    minCount = dataframe['Type'].value_counts()
    types = set(dataframe['Type'])
    newSet = pd.DataFrame()
    print(minCount)
    for i in types:
        query = 'Type in ' + '['+ '"' + i + '"' +  ']'
        newSet = newSet.append(dataframe.query(query)[:min], ignore_index = True)
    return newSet

data = getCleanData()
dataBarr = barrClasses(200,data)

equalset = equalizeClasses(250,dataBarr)
print(equalset.shape)


