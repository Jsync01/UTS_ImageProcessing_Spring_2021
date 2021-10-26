import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from shared import log, loadData

# Convoluted Neural Network
def cnn():
    x_train, x_test, y_train, y_test = loadData()
    log("Running Convoluted Neural Network Model")
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

    RAWmodel.add(Dense(47))

    RAWmodel.add(Activation('sigmoid'))

    RAWmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    log("Compiled Model")

    RAWmodel.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1)

    RAWmodel.evaluate(x_test, y_test)
    return