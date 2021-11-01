from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np
import cv2
from skimage import feature

from shared import log, loadData

def knn():
    log("Running K - nearest neighbours model")
    
    #retrieve training data
    x_train, x_test, y_train, y_test = loadData()

    print(np.shape(x_train))
    print(np.shape(x_test))

    samples, x, y, u = x_train.shape
    Xtrain = x_train.reshape((samples,x*y))
    t_samples, t_x, t_y, u = x_test.shape
    Xtest = x_test.reshape((t_samples,t_x*t_y))

    print(np.shape(Xtrain))
    print(np.shape(Xtest))

    #Neirest neighbours classifier with 7 neighbours
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(Xtrain, y_train.ravel())

    #train accuracy
    acc_train = clf.score(Xtrain, y_train)
    log(f'train set accuracy: {acc_train}')

    #test accuracy
    acc_test = clf.score(Xtest, y_test)
    log(f'Test set accuracy: {acc_test}')

    return

# Histogram of Oriented Gradient
def hogknn():
    log("Running Histogram of Oriented Gradients with knn")
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

    print(np.shape(HOGdata_train))
    print(np.shape(HOGdata_test))

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(HOGdata_train, HOGlabels_train.ravel())

    acc_train = clf.score(HOGdata_train, HOGlabels_train)
    log(f'train set accuracy: {acc_train}')

    acc_test = clf.score(HOGdata_test, HOGlabels_test)
    log(f'test set accuracy:  {acc_test}')

    return

# Local Binary Patterns
def lbpknn():
    x_train, x_test, y_train, y_test = loadData()
    log("Running Local Binary Patterns Model")
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

    LBPdata_train = np.asarray(LBPdata_train)
    LBPlabels_train = np.asarray(LBPlabels_train)
    LBPdata_test = np.asarray(LBPdata_test)
    LBPlabels_test = np.asarray(LBPlabels_test)

    print(np.shape(LBPdata_train))
    print(np.shape(LBPdata_test))

    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(LBPdata_train, LBPlabels_train.ravel())

    acc_train = clf.score(LBPdata_train, LBPlabels_train)
    log(f'train set accuracy: {acc_train}')

    acc_test = clf.score(LBPdata_test, LBPlabels_test)
    log(f'test set accuracy:  {acc_test}')

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