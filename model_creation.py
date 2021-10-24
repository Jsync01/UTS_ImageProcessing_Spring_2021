import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import math
import datetime
from skimage import feature
from skimage.color import rgb2gray
import sys


def unison_shuffled_copies(a, b):
    """Returns 2 unison shuffled copies of array a and b"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def log(logstr):
    """Prints logstr to console with current time"""
    print(datetime.datetime.now().isoformat() + " " + logstr)


def main():
    modelType = sys.argv[-1]
    select_model(modelType)


def load_data():
    # LOADING DATA
    log("Loading data...")
    images = np.load("nist_images_32x32.npy")
    labels = np.load("nist_labels_32x32.npy")
    log("Data loaded... Shuffling...")
    images, labels = unison_shuffled_copies(images, labels)
    log("Shuffled!")
    split = math.ceil(len(images) * 0.7)
    train_imgs = images[:split]
    train_labels = labels[:split]
    test_imgs = images[split:]
    test_labels = labels[split:]
    x_train = train_imgs/255.0
    x_test = test_imgs/255.0
    y_train = train_labels.astype('float64')
    y_test = test_labels.astype('float64')
    log("Performed train-test split")
    return x_train, x_test, y_train, y_test


def select_model(modelType):
    if(modelType == 'raw'):
        raw_model()
    elif(modelType == 'lbp'):
        lbp_model()
    elif(modelType == 'hog'):
        hog_model()
    else:
        print('Model type not recognised')
        return


def raw_model():
    x_train, x_test, y_train, y_test = load_data()
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


def lbp_model():
    x_train, x_test, y_train, y_test = load_data()
    desc = LocalBinaryPatterns(24, 8)
    LBPdata_train = []
    LBPlabels_train = []
    LBPdata_test = []
    LBPlabels_test = []

    x_train = np.array(x_train[:,:,:,0])
    x_test = np.array(x_test[:,:,:,0])

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

    LBPvalid_images = LBPdata_train[:5000] / 255.0
    LBPvalid_labels = LBPlabels_train[:5000]

    LBPtrain_images = LBPdata_train[5000:] / 255.0
    LBPtrain_labels = LBPlabels_train[5000:]
    
    log("Extracted test data")
    
    LBPmodel = Sequential()
    LBPmodel.add(Dense(512))
    LBPmodel.add(Dropout(0.2))

    LBPmodel.add(Dense(47))

    LBPmodel.add(Activation('sigmoid'))

    LBPmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    log("Compiled Model")

    print(np.shape(LBPdata_train))
    print(np.shape(LBPdata_test))
    print(np.shape(LBPlabels_train))
    print(np.shape(LBPlabels_test))

    LBPmodel.fit(LBPtrain_images, LBPtrain_labels, batch_size=128, epochs=10, validation_data=(LBPvalid_images, LBPvalid_labels))

    LBPmodel.evaluate(LBPdata_test, LBPlabels_test)


def hog_model():
    x_train, x_test, y_train, y_test = load_data()
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

    HOGANNdata_train = np.asarray(HOGdata_train)
    HOGANNlabels_train = np.asarray(HOGlabels_train)
    HOGANNdata_test = np.asarray(HOGdata_test)
    HOGANNlabels_test = np.asarray(HOGlabels_test)

    HOGANNdata_train = HOGANNdata_train/255.0
    HOGANNdata_test = HOGANNdata_test/255.0

    HOGmodel = Sequential()
    HOGmodel.add(Dense(512))
    HOGmodel.add(Dropout(0.2))

    HOGmodel.add(Dense(47))

    HOGmodel.add(Activation('sigmoid'))

    HOGmodel.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
    log("Compiled Model")


    HOGmodel.fit(HOGANNdata_train, HOGANNlabels_train, batch_size=128, epochs=10, validation_split=0.1)

    HOGmodel.evaluate(HOGANNdata_test, HOGANNlabels_test)


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


if __name__ == "__main__":
    main()