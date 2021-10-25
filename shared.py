import datetime, math
import numpy as np

def log(logstr):
    """Prints logstr to console with current time"""
    print(datetime.datetime.now().isoformat() + " " + logstr)


def loadData():
    # LOADING DATA
    log("Loading data...")
    images = np.load("nist_images_32x32.npy")
    labels = np.load("nist_labels_32x32.npy")
    log("Data loaded... Shuffling...")

    assert len(images) == len(labels)
    p = np.random.permutation(len(images))
    images = images[p]
    labels = labels[p]
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
    log("Performed train-test split with 0.7/0.3")
    return x_train, x_test, y_train, y_test