import sys, time

modelType = sys.argv[-1]

# Entrypoint for all model selection input
if (modelType == 'cnn'):
    cnnType = input("Which model to use for cnn [normal] [alexnet] [lenet] ")
    if (cnnType == 'normal'):
        from cnn import cnn
        startTime = time.time()
        cnn()
        print(f"Run time: {time.time() - startTime}")
    elif (cnnType == 'alexnet'):
        from cnn import adapted_alexnet_cnn
        startTime = time.time()
        adapted_alexnet_cnn()
        print(f"Run time: {time.time() - startTime}")
    elif (cnnType == 'lenet'):
        from cnn import lenet_cnn
        startTime = time.time()
        lenet_cnn()
        print(f"Run time: {time.time() - startTime}")
    else:
        print("Incorrect choice, try again.")
    
elif (modelType == "tree"):
    from tree import decisionTree
    startTime = time.time()
    decisionTree()
    print(f"Run time: {time.time() - startTime}")

elif (modelType == 'ann'):
    annType = input("Which model to use for ann [normal] [lbp] [hog] ")
    if (annType == 'normal'):
        from ann import ann
        startTime = time.time()
        ann()
        print(f"Run time: {time.time() - startTime}")
    elif (annType == 'lbp'):
        from ann import lbpann
        startTime = time.time()
        lbpann()
        print(f"Run time: {time.time() - startTime}")
    elif (annType == 'hog'):
        from ann import hogann
        startTime = time.time()
        hogann()
        print(f"Run time: {time.time() - startTime}")
    else:
        print("Incorrect choice, try again.")

elif (modelType == 'knn'):
    knnType = input("Which model to use for knn [normal] [lbp] [hog] ")
    if (knnType == 'normal'):
        from knn import knn
        startTime = time.time()
        knn()
        print(f"Run time: {time.time() - startTime}")
    elif (knnType == 'lbp'):
        from knn import lbpknn
        startTime = time.time()
        lbpknn()
        print(f"Run time: {time.time() - startTime}")
    elif (knnType == 'hog'):
        from knn import hogknn
        startTime = time.time()
        hogknn()
        print(f"Run time: {time.time() - startTime}")
    else:
        print("Incorrect choice, try again.")

elif (modelType == 'lbp'):
    from other import lbpModel
    startTime = time.time()
    lbpModel()
    print(f"Run time: {time.time() - startTime}")

elif (modelType == 'hog'):
    from other import hogModel
    startTime = time.time()
    hogModel()
    print(f"Run time: {time.time() - startTime}")

else:
    print('Model type not recognised')