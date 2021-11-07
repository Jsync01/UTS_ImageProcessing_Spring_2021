from shared import log, loadData
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from skimage import feature


# Aritificial Neural Network
def ann():
    x_train, x_test, y_train, y_test = loadData()
    
    train_images  = np.array(x_train, dtype= 'uint8') / 255.0
    train_labels  = np.array(y_train, dtype= 'uint8')

    test_images = np.array(x_test, dtype= 'uint8') / 255.0
    test_labels = np.array(y_test)
    print(np.shape(train_images))
    print(np.shape(test_images))
    
# Defining dense layers
    modelANN = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=[32,32]),
                                    tf.keras.layers.Dense(4096, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(4096, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(47, activation=tf.nn.softmax)])

    modelANN.compile(optimizer = tf.optimizers.Adam(),loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
# Defining the amount of epochs that will be calculated currently we have 10
    H=modelANN.fit(train_images, train_labels, epochs=10, validation_split=(0.1))

    modelANN.evaluate(test_images, test_labels)



    return

    

# Local Binary Patterns
def lbpann():
    x_train, x_test, y_train, y_test = loadData()
    log("Running Local Binary Patterns Model")
    desc = LocalBinaryPatterns(24, 8)
    LBPdata_train = []
    LBPlabels_train = []
    LBPdata_test = []
    LBPlabels_test = []

    x_train = np.array(x_train[:,:,:,0])
    x_test = np.array(x_test[:,:,:,0])
# EXtracting LBP features
    log("Extracting features from training dataset...")
    for img_index in range(len(x_train)):
        image = x_train[img_index]
        hist = desc.LBPfeatures(image)
    
        LBPlabels_train.append(y_train[img_index])
        LBPdata_train.append(hist)
    
    log("Extracted train data")
    for img_index in range(len(x_test)):
        
        image = x_test[img_index]
        hist = desc.LBPfeatures(image)
    
        LBPlabels_test.append(y_test[img_index])
        LBPdata_test.append(hist)

    LBPdata_train = np.asarray(LBPdata_train)
    LBPlabels_train = np.asarray(LBPlabels_train)
    LBPdata_test = np.asarray(LBPdata_test)
    LBPlabels_test = np.asarray(LBPlabels_test)

    
    LBPdata_train = LBPdata_train / 255.0
    LBPdata_test = LBPdata_test / 255.0
    
    log("Extracted test data")
    
    LBPmodel = Sequential()

    LBPmodel.add(Dense(512))

    LBPmodel.add(Dropout(0.2))
    LBPmodel.add(Dense(256))

    LBPmodel.add(Dense(47))

    LBPmodel.add(Activation('sigmoid'))
# Inserting LBP features into ANN classifier 
    LBPmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    log("Compiled Model")

    print(np.shape(LBPdata_train))
    print(np.shape(LBPdata_test))
    print(np.shape(LBPlabels_train))
    print(np.shape(LBPlabels_test))

    LBPmodel.fit(LBPdata_train, LBPlabels_train, batch_size=128, epochs=10, validation_split=0.1)
    LBPmodel.evaluate(LBPdata_test, LBPlabels_test)

    return


# Histogram of Oriented Gradient
def hogann():
    
    log("Running Histogram of Oriented Gradients Model")
    x_train, x_test, y_train, y_test = loadData()
    log("Extracting features from training dataset...")
    HOGdata_train = []
    HOGlabels_train = []
    HOGdata_test = []
    HOGlabels_test = []

    #Extracting features from train dataset
    for img_index in range(len(x_train)):
        image = x_train[img_index]
        
        H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2,2),
                        transform_sqrt=True, block_norm="L2-Hys") # Complete the code 
        
        HOGdata_train.append(H) 
        HOGlabels_train.append(y_train[img_index])
    
    log("Extracted train data")

    #Extracting features from test dataset
    for img_index in range(len(x_test)):
        image = x_test[img_index]
        
        H = feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2,2),
                        transform_sqrt=True, block_norm="L2-Hys") # Complete the code 
        
        HOGdata_test.append(H) 
        HOGlabels_test.append(y_test[img_index])

    log("Extracted test data")

    HOGdata_train = np.asarray(HOGdata_train)
    HOGlabels_train = np.asarray(HOGlabels_train)
    HOGdata_test = np.asarray(HOGdata_test)
    HOGlabels_test = np.asarray(HOGlabels_test)

    HOGdata_train = HOGdata_train / 255.0
    HOGdata_test = HOGdata_test / 255.0

    HOGmodel = Sequential()
    HOGmodel.add(Dense(512))

    HOGmodel.add(Dropout(0.2))
    HOGmodel.add(Dense(256))

    HOGmodel.add(Dense(47))

    HOGmodel.add(Activation('sigmoid'))

    HOGmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    log("Compiled Model")

    HOGmodel.fit(HOGdata_train, HOGlabels_train, batch_size=128, epochs=10, validation_split=0.1)
    HOGmodel.evaluate(HOGdata_test, HOGlabels_test)

    return


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius

    def LBPfeatures(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        return hist