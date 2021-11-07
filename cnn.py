import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
import numpy as np

from shared import log, loadData

# Convoluted Neural Network
def cnn():
    log("Running Convoluted Neural Network Model")
    x_train, x_test, y_train, y_test = loadData() # Loading in data

    # Ensuring output is in correct format
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Normalising data so that it ranges from 0-1 rather than 0-255
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Creating model architecture by adding layers individually
    log("Running Convolutional Neural Network Model")
    RAWmodel = Sequential()
    RAWmodel.add(Conv2D(16, (3,3), input_shape = x_train.shape[1:]))
    RAWmodel.add(Activation("relu"))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Conv2D(32, (3,3)))
    RAWmodel.add(Activation("relu"))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Conv2D(64, (3,3)))
    RAWmodel.add(Activation("relu"))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Flatten())
    RAWmodel.add(Dense(512))

    RAWmodel.add(Dropout(0.2))
    RAWmodel.add(Dense(256))

    RAWmodel.add(Dense(47))

    RAWmodel.add(Activation('sigmoid'))

    # Compiling the specified layers into a model
    RAWmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    log("Compiled Model")
    
    # Fitting the model to the train dataset and using 20% of the train data as validation
    RAWmodel.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1, verbose=1)

    # Evaluating the model on the test dataset
    RAWmodel.evaluate(x_test, y_test)
    return

# AlexNet Convoluted Neural Network Implementation
def adapted_alexnet_cnn():
    log("Running AlexNet Convoluted Neural Network Model")
    x_train, x_test, y_train, y_test = loadData() # Loading in data

    # Ensuring output is in correct format
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Normalising data so that it ranges from 0-1 rather than 0-255
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Creating model architecture by adding layers individually
    # Uses the same layer structure as the AlexNet model with different parameters
    log("Running Alex Net CNN Model")
    RAWmodel = Sequential()
    RAWmodel.add(Conv2D(24, (5,5), input_shape = x_train.shape[1:], padding='same', activation='relu'))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Conv2D(64, (3,3), padding='same', activation='relu'))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Conv2D(96, (3,3), padding='valid', activation='relu'))
    RAWmodel.add(Conv2D(96, (3,3), padding='valid', activation='relu'))
    RAWmodel.add(Conv2D(64, (3,3), padding='valid', activation='relu'))
    RAWmodel.add(MaxPooling2D(pool_size=(2,2)))

    RAWmodel.add(Flatten())
    
    RAWmodel.add(Dense(4096, activation='relu'))
    RAWmodel.add(Dense(4096, activation='relu'))

    RAWmodel.add(Dense(47, activation='softmax'))

    # Compiling the model
    RAWmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    log("Compiled Model")

    # Fitting the model to the train dataset using 10% of the data as validation
    RAWmodel.fit(x_train, y_train, batch_size=64, epochs=25, validation_split=0.1)

    RAWmodel.evaluate(x_test, y_test)
    return

# LeNet Convoluted Neural Network Implementation
def lenet_cnn():
    log("Running LeNet Convoluted Neural Network Model")
    x_train, x_test, y_train, y_test = loadData() # Loading in data

    # Ensuring output is in correct format
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    # Normalising data so that it ranges from 0-1 rather than 0-255
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Creating model architecture by adding layers individually
    # Uses the same layer structure as the LeNet model with different parameters
    log("Running Lenet CNN Model")
    RAWmodel = Sequential()
    RAWmodel.add(Conv2D(6, (5,5), strides=(1,1), activation='tanh', input_shape = x_train.shape[1:], padding='same'))
    RAWmodel.add(AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    RAWmodel.add(Conv2D(16, (5,5), strides=(1,1), activation='tanh', padding='valid'))
    RAWmodel.add(AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    RAWmodel.add(Conv2D(120, (5,5), strides=(1,1), activation='tanh', padding='valid'))

    RAWmodel.add(Flatten())
    
    RAWmodel.add(Dense(84, activation='tanh'))

    RAWmodel.add(Dense(47, activation='softmax'))

    # Compiling the model
    RAWmodel.compile(loss="sparse_categorical_crossentropy", optimizer='SGD', metrics=['accuracy'])
    log("Compiled Model")

    # Fitting the model on the train dataset using 10% of the data as validation
    RAWmodel.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

    # Evaluating the model on the train dataset
    RAWmodel.evaluate(x_test, y_test)
    return