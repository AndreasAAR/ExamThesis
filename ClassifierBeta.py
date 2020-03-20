from numpy import loadtxt

import Datamanagement
import Datamanagement as DM
import Graphs as Grap
#import StatMethods as Stat
from sklearn.neural_network import MLPClassifier
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.callbacks import EarlyStopping

#TODO
# 1. Make this file into a classifier method library!
# 2. Run testing and training separately
# 3. Implement Pre-training and saving model
# 4 .Save model to file
# 5. Save results file
# 6. Make table of results and accuracy barplot

#pre-training with keras
#https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/

#Basic
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
#Convergence
#https://stackoverflow.com/questions/53478622/stop-keras-training-when-the-network-has-fully-converge
#Saving and loading
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

#Set is already saved, equalized, 299,299 binary class problem,
#Keep at false unless when creating new sets
#Prepp datasets, save to file, saves time!
if(False):
    data = DM.getCleanData()
    data = DM.barrClasses(258,data)
    data = DM.equalizeClasses(258, data)
    multiData = data
    print(DM.getFrequencies(multiData))
    binaryData = data.loc[data["Type"].isin(["Neuroepithelial tumours", "B-cell lymphoma"])]
    DM.saveToFile("Datasets/binaryData.csv",binaryData)
    DM.saveToFile("Datasets/multiData.csv", multiData)

binaryData = pd.read_csv("Datasets/binaryData.csv")
multiData = pd.read_csv("Datasets/multiData.csv")
#print(DM.getFrequencies(binaryData))
#print(DM.getFrequencies(multiData))
#print(binaryData.head())


testPercentage = 0.3
trainingAndTestSet = DM.trainingTestData(multiData,0.3)
multiTrainingSet = trainingAndTestSet.get('training')
multiTestSet = trainingAndTestSet.get("test")

binaryTrainingAndTestSet = DM.trainingTestData(binaryData,0.3)
binaryTrainingSet = binaryTrainingAndTestSet.get('training')
binaryTestSet = binaryTrainingAndTestSet.get("test")

binaryInputAndClass = DM.inputClassSplitter(binaryTrainingSet)
binaryTrainingInputs = binaryInputAndClass.get("inputs")
binaryTrainingClasses = binaryInputAndClass.get("classes")

binaryInputAndClass = DM.inputClassSplitter(binaryTrainingSet)
binaryTrainingInputs = binaryInputAndClass.get("inputs")
binaryTrainingClasses = binaryInputAndClass.get("classes")

#Needs to be made generic, is the variables that are inputted
input = binaryData.shape[1]-1
#Thesis Peng related variables
patience = 100 #When to stop trying to improve, when accuracy doesent improve
layerNum = 2  #Make deep, 5 layers later
layerNodes = 4 #turn into 1024 later

#Convergence!
overfitCallback = EarlyStopping(monitor='loss', min_delta=0, patience = patience)

#Training model on training set
if(False):
    model = Sequential()
    model.add(Dense(layerNodes, input_dim=input, activation='relu'))
    for i in range(layerNum):
        model.add(Dense(layerNodes, activation='relu')) #add five layers!
    model.add(Dense(1, activation='softmax'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=1000, batch_size=10, callbacks=[overfitCallback])
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))
    #Implementing convergence




