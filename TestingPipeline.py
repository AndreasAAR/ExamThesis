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

#NOTE Equalize actually normalises AND topbarr.
def testPipe(name, topBarr,botBarr,equalizeLevel,dirty = False):
    data = DM.getCleanData()
    try:
        modifiedData = pd.read_csv("Datasets/" + name + ".csv")
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        modifiedData = DM.barrClasses(topBarr, data)
        if (botBarr != None):
            modifiedData = DM.lowBarrClasses(botBarr, data)
        modifiedData = DM.equalizeClasses(equalizeLevel, modifiedData)
        DM.saveToFile("Datasets/" + name + ".csv", modifiedData)

    modifiedData = pd.read_csv("Datasets/"+name+".csv")
    print(DM.getFrequencies(modifiedData))
    
    #Make training and test split
    if(True):
       testPercentage = 0.3
       dataTrainingAndTestSet = DM.trainingTestData(modifiedData, testPercentage)
       dataTrainingSet = dataTrainingAndTestSet.get('training')
       dataTestSet = dataTrainingAndTestSet.get("test")
    
    patience = 100  # When to stop trying to improve, when accuracy doesent improve
    layerNum = 5  # Make deep, 5 layers later
    layerNodes = 1024  # turn into 1024 later
    learningRate = 0.01  #Correct with peng

    #Choosing whole data for PT if dirty
    if(dirty == True):
        dataPreTrainSet = DM.getPreTrainSet(data)
    else:
        dataPreTrainSet = DM.getPreTrainSet(modifiedData)

    dataPreTrainSetSplit = DM.inputClassSplitter(dataPreTrainSet)
    dataPreTrainInputs = dataPreTrainSetSplit.get('inputs')
    dataPreTrainClasses = dataPreTrainSetSplit.get('classes')

    if (dirty == False):
        #Pretraining for dataclass
        try:
            #If no previous file create and train!
            dataPreTrainModel = load_model("ClassifierWeights/" +"PTpart"+ name + ".h5")
        except:
            preTrainModel = CH.trainModel(dataPreTrainInputs,dataPreTrainClasses,patience,layerNum,layerNodes,learningRate)
            preTrainModel.save("ClassifierWeights/"+"PTpart"+name + ".h5")
            dataPreTrainModel = load_model("ClassifierWeights/"+"PTpart"+ name + ".h5")
    else:
        try:
            #If no previous file create and train!
            dataPreTrainModel = load_model("ClassifierWeights/" +"DirtyPTModel"+".h5")
        except:
            preTrainModel = CH.trainModel(dataPreTrainInputs,dataPreTrainClasses,patience,layerNum,layerNodes,learningRate)
            preTrainModel.save("ClassifierWeights/" +"DirtyPTModel"+".h5")
            dataPreTrainModel = load_model("ClassifierWeights/" +"DirtyPTModel"+".h5")

    #For testing
    dataTrainingInputAndClass = DM.inputClassSplitter(dataTrainingSet)
    dataTrainingInputs = dataTrainingInputAndClass.get("inputs")
    dataTrainingClasses = dataTrainingInputAndClass.get("classes")
    dataTestingInputAndClass = DM.inputClassSplitter(dataTestSet)
    dataTestingInputs = dataTestingInputAndClass.get("inputs")
    dataTestingClasses = dataTestingInputAndClass.get("classes")

    #Accuracy testing with PT
    try:
        mPTM = load_model("ClassifierWeights/"+"PT"+  name + ".h5")
    except:
        #data with PT
        mPTM =  CH.trainModel(dataTrainingInputs,dataTrainingClasses,patience,layerNum,layerNodes,learningRate,dataPreTrainModel)
        mPTM.save("ClassifierWeights/"+"PT"+ name + ".h5")
        mPTM = load_model("ClassifierWeights/"+"PT" + name + ".h5")



    try:
        mM = load_model("ClassifierWeights/"+ name + ".h5")
    except:
        #data without PT
        mM = CH.trainModel(dataTrainingInputs,dataTrainingClasses,patience,layerNum,layerNodes,learningRate)
        mM.save("ClassifierWeights/"+ name + ".h5")
        mM = load_model("ClassifierWeights/"+ name + ".h5")

    _, accuracy = mPTM.evaluate(dataTestingInputs, dataTestingClasses)
    print('Accuracy mPT: %.2f' % (accuracy * 100))

    _,accuracy = mM.evaluate(dataTestingInputs,dataTestingClasses)
    print('Accuracy mM: %.2f' % (accuracy * 100))
