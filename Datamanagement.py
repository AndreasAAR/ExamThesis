import numpy as np, numpy
import pandas as pd
import sklearn as sk

#TODO
# Make sure all data-management methods
# or repeated code ends up here

###Clean Table extraction from Jaakkos set,
# Also adds the correct class labels
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

def getPreTrainSet(data):
    PreTrainNormal = data
    PreTrainNormal.loc[:, 'Type'] = '1'
    PreTrainShuffled = shuffleCollumns(data.copy())
    PreTrainShuffled.loc[:, 'Type'] = '0'
    PreTrainSet = pd.concat([PreTrainShuffled, PreTrainNormal])
    return PreTrainSet

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

#data = getCleanData()

#Frequencies creates a 2-column CLASS, #REPETITIONS, dataset.
def getFrequencies(data):
    frequencies = data["Type"].value_counts()
    return frequencies

#Keeps the classes that passes the lowest level of frequency,
#Does not adjust number of entries however!
def barrClasses(lowestRep, dataframe):
    dataframe = dataframe.groupby('Type').filter(lambda x : len(x) >= lowestRep)
    return dataframe

def lowBarrClasses(highestRep, dataframe):
    dataframe = dataframe.groupby('Type').filter(lambda x : len(x) <= highestRep)
    return dataframe

#Saves the file to a filepath provided, does not add rownumbers, avoid strange extra columns when reading again!
def saveToFile(path, data):
    data.to_csv(path,  encoding='utf-8', index=False)

#Equalizes all the entries to the number of classes provided
#Not guaranteed to work if not using the smallest class in the dataset
def equalizeClasses(lowestRep, dataframe):
    dataframe = barrClasses(lowestRep,dataframe)
    #min = dataframe['Type'].value_counts().min()
    #minCount = dataframe['Type'].value_counts()
    types = set(dataframe['Type'])
    newSet = pd.DataFrame()
    for i in types:
        query = 'Type in ' + '['+ '"' + i + '"' +  ']'
        newSet = newSet.append(dataframe.query(query)[:lowestRep], ignore_index = True)
    return newSet

#Shuffles the values in the collumn of the set
def shuffleCollumns(dataframe):
    for i in dataframe.columns:
        if i != 0:
            shuffledCol = sk.utils.shuffle(dataframe[i])
            shuffledCol.reset_index(inplace = True,drop = True)
            dataframe.reset_index(inplace=True, drop=True)
            dataframe[i] = shuffledCol
    return dataframe



#Returns array with training and testing
def trainingTestData(data,testPercentage):
    classes = data.Type.unique()
    testPercentage = 0.3
    testReturn = pd.DataFrame
    trainingReturn = pd.DataFrame
    testFrames = []
    trainingFrames = []
    for c in classes:
       classSet = data.loc[data["Type"].isin([c])]
       testGroup = classSet.iloc[0: int(classSet.shape[1]* testPercentage),:]
       trainingGroup = classSet.iloc[int(classSet.shape[1] * testPercentage):classSet.shape[1], :]
       trainingFrames.append(trainingGroup)
       testFrames.append(testGroup)
       testReturn = pd.concat(testFrames)
       trainingReturn = pd.concat(trainingFrames)
    returnDict = {
        "training": trainingReturn,
        "test": testReturn,
    }
    return returnDict

#Splits data into a input variable dataframe, and integer class vector
def inputClassSplitter(data):
    X = data.iloc[:, 0:data.shape[1] - 1]  # Should not contain classses
    y = data.iloc[:, data.shape[1] - 1]  # Contains the classes
    classes = data.Type.unique()
    classNum = 0
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    inputClassDict = {"inputs": X, "classes": dummy_y}
    return inputClassDict


#listTest = [0,1,0,1]
#print(sk.utils.shuffle(listTest))

#equalset = equalizeClasses(20,dataBarr)
#print(equalset['Type'].value_counts())


#data = {"a": [1,1,0,0], "b": [1,1,0,0],"c": [1,1,0,0]}
#dataFrame = pd.DataFrame(data)
#dataFrame = shuffleCollumns(dataFrame)


#print(dataFrame)


