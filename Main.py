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
import ClassifierHelper as CH
from keras.models import load_model

#TODO
# 1. Make this file into a classifier method library!
# 2. Run testing and training separately
# 3. Implement Pre-training and saving model
# 4 .Save model to file
# 5. Save results file
# 6. Make table of results and accuracy barplot

#Based on:
#https://github.com/superRookie007/supervised_pretraining/blob/master/train_uci.py

#pre-training with keras
#https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/
#Set is already saved, equalized, 299,299 binary class problem,
#Keep at false unless when creating new sets
#Prepp datasets, save to file, saves time!
if(False):
    data = DM.getCleanData()
    data = DM.barrClasses(258,data)
    data = DM.equalizeClasses(258, data)
    multiData = data
    binaryData = data.loc[data["Type"].isin(["Neuroepithelial tumours", "B-cell lymphoma"])]
    DM.saveToFile("Datasets/binaryData.csv",binaryData)
    DM.saveToFile("Datasets/multiData.csv", multiData)

binaryData = pd.read_csv("Datasets/binaryData.csv")
multiData = pd.read_csv("Datasets/multiData.csv")

if(True):
    testPercentage = 0.3
    multiTrainingAndTestSet = DM.trainingTestData(multiData,0.3)
    multiTrainingSet = multiTrainingAndTestSet.get('training')
    multiTestSet = multiTrainingAndTestSet.get("test")

    binaryTrainingAndTestSet = DM.trainingTestData(binaryData,0.3)
    binaryTrainingSet = binaryTrainingAndTestSet.get('training')
    binaryTestSet = binaryTrainingAndTestSet.get("test")

patience = 100  # When to stop trying to improve, when accuracy doesent improve
layerNum = 5  # Make deep, 5 layers later
layerNodes = 1024  # turn into 1024 later
learningRate = 0.01  #Correct with peng


#Make pretrain-set
binaryPreTrainSet = DM.getPreTrainSet(binaryData)
DM.getFrequencies(binaryPreTrainSet)
print(DM.getFrequencies(binaryPreTrainSet))

binaryPreTrainSetSplit = DM.inputClassSplitter(binaryPreTrainSet)
binaryPreTrainInputs = binaryPreTrainSetSplit.get('inputs')
binaryPreTrainClasses = binaryPreTrainSetSplit.get('classes')

#Pretrain for binary
if(False):
    #Pretraining a model
    preTrainModel = CH.trainModel(binaryPreTrainInputs,binaryPreTrainClasses,patience,layerNum,layerNodes,learningRate)
    preTrainModel.save("ClassifierWeights/binaryPreTrain.h5")
binaryPreTrainModel = load_model("ClassifierWeights/binaryPreTrain.h5")

#Make multi pretrainset
multiPreTrainSet = DM.getPreTrainSet(multiData)
multiPreTrainSetSplit = DM.inputClassSplitter(multiPreTrainSet)
multiPreTrainInputs = multiPreTrainSetSplit.get('inputs')
multiPreTrainClasses = multiPreTrainSetSplit.get('classes')

#Pretraining for multiclass
if(False):
    #Pretraining a model
    preTrainModel = CH.trainModel(multiPreTrainInputs,multiPreTrainClasses,patience,layerNum,layerNodes,learningRate)
    preTrainModel.save("ClassifierWeights/multiPreTrain.h5")
multiPreTrainModel = load_model("ClassifierWeights/multiPreTrain.h5")

#For binary testing
binaryTrainingInputAndClass = DM.inputClassSplitter(binaryTrainingSet)
binaryTrainingInputs = binaryTrainingInputAndClass.get("inputs")
binaryTrainingClasses = binaryTrainingInputAndClass.get("classes")
binaryTestingInputAndClass = DM.inputClassSplitter(binaryTestSet)
binaryTestingInputs = binaryTestingInputAndClass.get("inputs")
binaryTestingClasses = binaryTestingInputAndClass.get("classes")

#For multi testing
multiTrainingInputAndClass = DM.inputClassSplitter(multiTrainingSet)
multiTrainingInputs = multiTrainingInputAndClass.get("inputs")
multiTrainingClasses = multiTrainingInputAndClass.get("classes")
multiTestingInputAndClass = DM.inputClassSplitter(multiTestSet)
multiTestingInputs = multiTestingInputAndClass.get("inputs")
multiTestingClasses = multiTestingInputAndClass.get("classes")

#binary with PT
#CH.trainModel(binaryTrainingInputs,binaryTrainingClasses,patience,layerNum,layerNodes,learningRate,binaryPreTrainModel)
CH.trainModel(binaryTrainingInputs,binaryTrainingClasses,patience,layerNum,layerNodes,learningRate)

#Multi with PT
#CH.trainModel(multiTrainingInputs,multiTrainingClasses,patience,layerNum,layerNodes,learningRate,multiPreTrainModel)
#Multi without PT
CH.trainModel(multiTrainingInputs,multiTrainingClasses,patience,layerNum,layerNodes,learningRate)