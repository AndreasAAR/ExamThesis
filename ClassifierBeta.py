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

#pre-training with keras
#https://machinelearningmastery.com/how-to-improve-performance-with-transfer-learning-for-deep-learning-neural-networks/

#Basic
#https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
#Convergence
#https://stackoverflow.com/questions/53478622/stop-keras-training-when-the-network-has-fully-converge
#Saving and loading
#https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# Example NN
if(False):
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (y) variables
    X = dataset[:,0:8]
    y = dataset[:,8]
    # define the keras model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy*100))


#Set is already saved, equalized, 299,299 binary class problem
if(False):
    data = DM.getCleanData()
    print(data.shape)
    data = data.loc[data["Type"].isin(["Neuroepithelial tumours","B-cell lymphoma"])]
    data = DM.equalizeClasses(299, data)
    print(DM.getFrequencies(data)) #299,299
    print(data.shape)
    DM.saveToFile("Datasets/equal299.csv",data)
data = pd.read_csv("Datasets/equal299.csv")

if(True):
    #print(data.shape) #(544, 394)
    #Iloc like slice for pandas df
    X = data.iloc[:, 0:393] #Should not contain classses
    y = data.iloc[:, 393] #Contains the classes
    y = y.replace( {'Neuroepithelial tumours': '1', 'B-cell lymphoma': '0'})


input = 393
if(True ):
    model = Sequential()
    model.add(Dense(12, input_dim=input, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

