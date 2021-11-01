import sys

modelType = sys.argv[-1]
if(modelType == 'cnn'):
    cnnType = input("Which model to use for cnn [normal] [alexnet] [lenet] ")
    if(cnnType == 'normal'):
        from cnn import cnn
        cnn()
    elif(cnnType == 'alexnet'):
        from cnn import adapted_alexnet_cnn
        adapted_alexnet_cnn()
    elif(cnnType == 'lenet'):
        from cnn import lenet_cnn
        lenet_cnn()
    else:
        print("Incorrect choice, try again.")
    
elif(modelType == "tree"):
    from tree import decisionTree
    decisionTree()

elif(modelType == 'ann'):
    annType = input("Which model to use for ann [normal] [lbp] [hog] ")
    if(annType == 'normal'):
        from ann import ann
        ann()
    elif(annType == 'lbp'):
        from ann import lbpann
        lbpann()
    elif(annType == 'hog'):
        from ann import hogann
        hogann()
    else:
        print("Incorrect choice, try again.")

elif(modelType == 'knn'):
    knnType = input("Which model to use for knn [normal] [lbp] [hog] ")
    if(knnType == 'normal'):
        from knn import knn
        knn()
    elif(knnType == 'lbp'):
        from knn import lbpknn
        lbpknn()
    elif(knnType == 'hog'):
        from knn import hogknn
        hogknn()
    else:
        print("Incorrect choice, try again.")

elif(modelType == 'lbp'):
    from other import lbpModel
    lbpModel()

elif(modelType == 'hog'):
    from other import hogModel
    hogModel()

else:
    print('Model type not recognised')