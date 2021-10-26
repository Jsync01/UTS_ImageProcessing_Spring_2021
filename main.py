import sys

modelType = sys.argv[-1]
if(modelType == 'cnn'):
    from cnn import cnn
    cnn()
elif(modelType == "dbn"):
    from dbn import deepBeliefNetwork
    deepBeliefNetwork()
elif(modelType == "tree"):
    from tree import decisionTree
    decisionTree()
elif(modelType == 'lbp'):
    from other import lbpModel
    lbpModel()
elif(modelType == 'hog'):
    from other import hogModel
    hogModel()
elif(modelType == 'knn'):
    from knn import knn
    knn()
elif(modelType == 'lbpknn'):
    from knn import lbpknn
    lbpknn()
elif(modelType == 'hogknn'):
    from knn import hogknn
    hogknn()
else:
    print('Model type not recognised')