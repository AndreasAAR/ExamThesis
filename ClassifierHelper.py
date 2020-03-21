from keras.callbacks import EarlyStopping
from sklearn.neural_network import MLPClassifier
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import keras as KS



def trainModel(X_train, y_train, patience, layerNum, layerNodes, learningRate, preTrainModel = None):
    #Needs to be made generic, is the variables that are inputted
    input = X_train.shape[1]
    print(X_train.shape)
    num_classes = y_train.shape[1] #collumns, the same as all label types
    #Thesis Peng related variables
    #Convergence!
    overfitCallback = EarlyStopping(monitor='accuracy', min_delta=0, patience=100, verbose=1, mode='auto')
    #Training model on training set
    model = Sequential()
    model.add(Dense(input, input_dim=input, activation='relu'))
    for i in range(layerNum):
        model.add(Dense(layerNodes, activation='relu')) #add layers
    model.add(Dense(num_classes, activation='softmax'))  #should be softmax
    # compile the keras model
    KS.optimizers.Adam(lr=learningRate)

    #adding pretrainweights
    if (preTrainModel != None):
        layers_list = preTrainModel.layers
        print("layers list" , len(layers_list))
        for i in range(0,layerNum+2):
            if(num_classes == 2):
               model.layers[i].set_weights(layers_list[i].get_weights())
            if (num_classes != 2 and i < layerNum+1):  # Skip last layer for multiclass!
                model.layers[i].set_weights(layers_list[i].get_weights())
            print(i)


    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) #SGD is closest to grad desc in Peng
    # fit the keras model on the dataset
    model.fit(X_train, y_train, epochs=1000, batch_size=X_train.shape[0], callbacks=[overfitCallback])
    # evaluate the keras model
    _, accuracy = model.evaluate(X_train, y_train)
    print('Accuracy: %.2f' % (accuracy * 100))
    #Implementing convergence
    return model
