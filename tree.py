from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from shared import log, loadData

# Decision Tree Classifier
def decisionTree():
    x_train, x_test, y_train, y_test = loadData()
    log("Running Decision Tree Classifier Model")

    log("Converting 3d array into 2d")
    nTrainSamples, nTrainX, nTrainY, trainX = x_train.shape
    x_train = x_train.reshape((nTrainSamples,nTrainX*nTrainY))

    nTestSamples, nTestX, nTestY, testX = x_test.shape
    x_test = x_test.reshape((nTestSamples,nTestX*nTestY))

    model = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=25)

    log("Fitting model")
    model.fit(x_train, y_train)

    log(f'Accuracy Score on train data: {round(accuracy_score(y_true=y_train, y_pred=model.predict(x_train)),3)}')
    log(f'Accuracy Score on the test data: {round(accuracy_score(y_true=y_test, y_pred=model.predict(x_test)),3)}')

    return