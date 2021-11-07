from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from graphviz import Source
import os

from shared import log, loadData

# Ensure the path to graphviz is mapped
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Decision Tree Classifier
def decisionTree():
    log("Running Decision Tree Classifier Model")
    x_train, x_test, y_train, y_test = loadData() # Loading in data
    
    log("Converting 3d array into 2d")
    nTrainSamples, nTrainX, nTrainY, trainX = x_train.shape
    x_train = x_train.reshape((nTrainSamples,nTrainX*nTrainY)) # Convert train data into 2d

    nTestSamples, nTestX, nTestY, testX = x_test.shape
    x_test = x_test.reshape((nTestSamples,nTestX*nTestY)) # Convert test data into 2d

    model = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=25) # Instantiates the classifer

    log("Fitting model")
    model.fit(x_train, y_train) # Starts fitting the model

    # Exports a decision tree image
    log("Generating illustration")
    graph = Source(export_graphviz(model, out_file=None, filled=True, max_depth=2)) # Export dot file for tree
    graph.format="png"
    graph.render('tree', view=True) # Render image from dot file

    # Accuracy calculation with rounding
    log(f'Accuracy Score on train data: {round(accuracy_score(y_true=y_train, y_pred=model.predict(x_train)),3)}')
    log(f'Accuracy Score on the test data: {round(accuracy_score(y_true=y_test, y_pred=model.predict(x_test)),3)}')
    
    return