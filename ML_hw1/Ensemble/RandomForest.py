from DecisionTreeNumeric import *
import numpy as np

def subsample(dataset, labels):
    upper_bound = len(dataset)
    indices = np.random.choice(upper_bound, 1000, replace=False)
    X = np.take(dataset, indices, axis=0)
    y = np.take(labels, indices)
    return X,y

def predict(trees, X):
    y_pred = [tree.predict(X) for tree in trees]
    y_pred = np.array(y_pred)
    predictions = (1*np.sign(np.sum(y_pred, axis = 0))).astype(int)
    return predictions

def random_forest(X, attributes, Y, tree_num=100, feature_sample=None, sample=True):
    trees = []
    for i in range(tree_num):
        if sample:
            sample_x, sample_y = subsample(X,Y)
            tree = DecisionTreeClassifier(S=sample_x, attributes=attributes, labels=sample_y, max_depth=-1, feature_sample=feature_sample)
        else:
            tree = DecisionTreeClassifier(S=X, attributes=attributes, labels=Y, max_depth=-1, feature_sample=feature_sample)
        tree.id3()
        trees.append(tree)
    return trees
