import numpy as np, numpy
import pandas as pd, pandas
import matplotlib
import collections

import sklearn as sklearn

import StatMethods as sm
import statistics as stat
import Graphs as grap
import sklearn as sk


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


data = getCleanData()

frequencies = data["Type"].value_counts()
print(type(frequencies))


def equalizeClass(lowestRep):
    print("IM METHOD")





def barrClasses(lowestRep, dataframe):
    dataframe = dataframe.groupby('Type').filter(lambda x : len(x) > lowestRep)
    return dataframe

def equalizeClasses(lowestRep, dataframe):
    dataframe = barrClasses(lowestRep,dataframe)
    min = dataframe['Type'].value_counts().min()
    minCount = dataframe['Type'].value_counts()
    types = set(dataframe['Type'])
    newSet = pd.DataFrame()
    
    for i in types:
        query = 'Type in ' + '['+ '"' + i + '"' +  ']'
        newSet = newSet.append(dataframe.query(query)[:min], ignore_index = True)
    return newSet

def shuffleCollumns(dataframe):
    for i in dataframe.columns:
        shuffledCol = sk.utils.shuffle(dataframe[i])
        shuffledCol.reset_index(inplace = True,drop = True)
        dataframe.reset_index(inplace=True, drop=True)
        dataframe[i] = shuffledCol
    return dataframe


data = getCleanData()
dataBarr = barrClasses(300,data)


#listTest = [0,1,0,1]
#print(sk.utils.shuffle(listTest))

equalset = equalizeClasses(20,dataBarr)
#print(equalset['Type'].value_counts())


#SMOTE
data = {"a": [1,1,0,0], "b": [1,1,0,0],"c": [1,1,0,0]}
dataFrame = pd.DataFrame(data)
dataFrame = shuffleCollumns(dataFrame)


print(dataFrame)


