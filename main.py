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
else:
    print('Model type not recognised')