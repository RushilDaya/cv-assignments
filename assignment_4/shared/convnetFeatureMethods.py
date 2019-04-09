'''
The using the conv net for features requires the network training..
Following this is the feature extraction
'''
import numpy as np
from shared.configurationParser import retrieveConfiguration as rCon
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def _simpleSplit(data,labels,train_ratio=0.7):

    numSamples = data.shape[0]
    numTraining = int(train_ratio*numSamples)
    return data[0:numTraining,:,:,:],labels[0:numTraining],data[numTraining:-1,:,:,:],labels[numTraining:-1]

def _flatten(data):
    colorDepth = data.shape[3]
    flattened = np.zeros((data.shape[0],data.shape[1],data.shape[2]),dtype='float')
    for idx in range(colorDepth):
        flattened= flattened + data[:,:,:,idx]/3
    return flattened

def _modelArchitecture(inputShape, num_classes):
    '''
        this architecture was obtained from a tutorial at 
        https://towardsdatascience.com/build-your-own-convolution-neural-network-in-5-mins-4217c2cf964f
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(inputShape[1],inputShape[2],inputShape[3])))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def learnConvNet(data, labels, num_classes):
    epochs = rCon('CONVNET_TRAIN_EPOCHS')
    batch_size = rCon('CONVNET_BATCH_SIZE')
    
    x_train,y_train, x_validate,y_validate = _simpleSplit(data,labels,train_ratio=0.7)
    x_train = _flatten(x_train)
    x_validate = _flatten(x_validate)

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
    x_validate = x_validate.reshape(x_validate.shape[0],x_validate.shape[1],x_validate.shape[2],1)

    x_train = x_train.astype('float32')
    x_train = x_train/255
    x_validate = x_validate.astype('float32')
    x_validate = x_validate/255

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_validate = keras.utils.to_categorical(y_validate, num_classes)

    model = _modelArchitecture(x_train.shape,num_classes)

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_validate, y_validate))
    return model

def convNetFeatures(data,labels,model=None):
    flat_data = _flatten(data)
    flat_data = flat_data.reshape(flat_data.shape[0],flat_data.shape[1],flat_data.shape[2],1)
    flat_data = flat_data.astype('float32')
    flat_data = flat_data/255
    predictions = model.predict(flat_data)
    return predictions, labels

def singleImageConv(image, model):
    temp_image = image.reshape(1,image.shape[0],image.shape[1],image.shape[2])
    prediction, label = convNetFeatures(temp_image,None,model=model)
    return prediction
