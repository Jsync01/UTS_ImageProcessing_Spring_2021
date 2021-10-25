import sys

modelType = sys.argv[-1]
if(modelType == 'cnn'):
    from cnn import cnn
elif(modelType == "dbn"):
    from dbn import deepBeliefNetwork
elif(modelType == "tree"):
    from tree import decisionTree
elif(modelType == 'lbp'):
    from other import lbpModel
elif(modelType == 'hog'):
    from other import lbpModel
else:
    print('Model type not recognised')