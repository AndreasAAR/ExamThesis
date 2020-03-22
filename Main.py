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
import TestingPipeline as TP


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

#def testPipe(name, topBarr,botBarr,equalizeLevel,dirty = False):


#print(DM.getFrequencies(DM.getCleanData()))

isDirty = False
topBarr = 299
equalize = 50
botBarr = None

Class = "bin"
name = "dirty_" + str(isDirty) +  "_tB_" + str(topBarr) + "_eQ_" + str(equalize) + "_bb_"  + str(botBarr) + "_Class_" + str(Class)

print(name)
TP.testPipe(name, topBarr,botBarr,equalize,isDirty)

if(False):
    if(False):
        data = DM.getCleanData()
        print(DM.getFrequencies(data))
        data = DM.barrClasses(288,data)
        data = DM.equalizeClasses(288, data)
        multiData = data
        print(DM.getFrequencies(data))
        binaryData = data.loc[data["Type"].isin(["Neuroepithelial tumours", "B-cell lymphoma"])]
        print(DM.getFrequencies(multiData))
        DM.saveToFile("Datasets/binaryData.csv",binaryData)
        DM.saveToFile("Datasets/multiData.csv", multiData)

    binaryData = pd.read_csv("Datasets/binaryData.csv")
    multiData = pd.read_csv("Datasets/multiData.csv")
    print(DM.getFrequencies(multiData))
    print(DM.getFrequencies(binaryData))

    #Make training and test split
    if(False):
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


        #Make binary pretrain-set
        binaryPreTrainSet = DM.getPreTrainSet(binaryData)
        DM.getFrequencies(binaryPreTrainSet)

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

    #Accuracy testing
    if(True):
        if(False):
            #binary with PT
            bPTM = CH.trainModel(binaryTrainingInputs,binaryTrainingClasses,patience,layerNum,layerNodes,learningRate,binaryPreTrainModel)
            bPTM.save("ClassifierWeights/binaryPreTrainTrained.h5")
        bPTM = load_model("ClassifierWeights/binaryPreTrainTrained.h5")
        _,accuracy = bPTM.evaluate(binaryTestingInputs,binaryTestingClasses)
        print('Accuracy bPT: %.2f' % (accuracy * 100))

        if(False):
            #binary without PT
            bM = CH.trainModel(binaryTrainingInputs,binaryTrainingClasses,patience,layerNum,layerNodes,learningRate)
            bM.save("ClassifierWeights/binaryTrained.h5")
        bM = load_model("ClassifierWeights/binaryTrained.h5")
        _,accuracy = bM.evaluate(binaryTestingInputs,binaryTestingClasses)
        print('Accuracy bM: %.2f' % (accuracy * 100))

        if(False):
            #Multi with PT
            mPTM =  CH.trainModel(multiTrainingInputs,multiTrainingClasses,patience,layerNum,layerNodes,learningRate,multiPreTrainModel)
            mPTM.save("ClassifierWeights/multiPreTrainTrained.h5")
        mPTM = load_model("ClassifierWeights/multiPreTrainTrained.h5")
        _,accuracy = mPTM.evaluate(multiTestingInputs,multiTestingClasses)
        print('Accuracy mPT: %.2f' % (accuracy * 100))

        if(False):
        #Multi without PT
            mM = CH.trainModel(multiTrainingInputs,multiTrainingClasses,patience,layerNum,layerNodes,learningRate)
            mM.save("ClassifierWeights/multiTrained.h5")
        mM = load_model("ClassifierWeights/multiTrained.h5")
        _,accuracy = mM.evaluate(multiTestingInputs,multiTestingClasses)
        print('Accuracy mM: %.2f' % (accuracy * 100))


