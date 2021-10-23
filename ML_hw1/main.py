from Ensemble.AdaBoost import *
from Ensemble.BaggedTrees import *
from Ensemble.RandomForest import *
from LinearRegression.LMS import *
from DataPreprocess import *
from DecisionTreeNumeric import *
import argparse
import pickle

parser = argparse.ArgumentParser(description='Pipeline commandline argument')
parser.add_argument("--data_dir", type=str, default="bank/", help="The directory of the dataset.")
parser.add_argument("--model", type=str, default="DecisionTreeClassifier")
parser.add_argument("--feature_sample", type=int, default=None)
parser.add_argument("--data_sample", type=bool, default=None)
parser.add_argument("--num_trees", type=int, default=500)
parser.add_argument("--max_depth", type=int, default=-1)
parser.add_argument("--split_scheme", type=str, default="Information Gain")
parser.add_argument("--train_error", type=bool, default=True)
parser.add_argument("--test_error", type=bool, default=True)

args = parser.parse_args()

if __name__ == "__main__":
    data_dir = args.data_dir
    data_preprocessor = DataPreprocess(data_dir+"train.csv", data_dir+"test.csv", data_dir+"data-desc.txt")
    X_train, Y_train, attributes_value = data_preprocessor.create_data_set()
    X_test, Y_test, _ = data_preprocessor.create_data_set(is_train=False)
    model = args.model
    predictor = None
    print("Running model "+model)
    if model == "DecisionTreeClassifier":
        predictor = DecisionTreeClassifier(S=X_train, attributes=attributes_value, labels=Y_train, max_depth=args.max_depth,
                                      split_scheme=args.split_scheme, feature_sample=args.feature_sample)
    elif model == "BaggedTrees":
        predictor = bagging(X=X_train, attributes=attributes_value, Y=Y_train, tree_num=args.num_trees, sample=args.data_sample)
    elif model == "AdaBoost":
        predictor = AdaBoost()
        predictor.fit(X=X_train, attributes=attributes_value, y=Y_train, T=args.num_trees)
    elif model == "RandomForest":
        predictor = random_forest(X=X_train, attributes=attributes_value, Y=Y_train, tree_num=args.num_trees, feature_sample=args.feature_sample,
                       sample=args.data_sample)
    elif model == "LMS":
        predictor = LMS()
    else:
        print("Please choose one of the available models")
        exit()
    if args.train_error:
        if model == "LMS":
            score = LMS.predict(X_train, Y_train)
            print("Cost Function Value: ", score)
        elif model == "AdaBoost":
            y_pred = predictor.predict(X_train)
            error = sum(np.not_equal(Y_train, y_pred)) / len(Y_train)
            print("Train error: ", error)
        else:
            y_pred = predict(predictor, X_train)
            error = sum(np.not_equal(Y_train, y_pred)) / len(Y_train)
            print("Train error: ", error)
    if args.test_error:
        if model == "LMS":
            score = LMS.predict(X_test, Y_test)
            print("Cost Function Value: ", score)
        elif model == "AdaBoost":
            y_pred = predictor.predict(X_test)
            error = sum(np.not_equal(Y_test, y_pred)) / len(Y_test)
            print("Train error: ", error)
        else:
            y_pred = predict(predictor, X_test)
            error = sum(np.not_equal(Y_test, y_pred)) / len(Y_test)
            print("Train error: ", error)

    file_to_store = open(model+".pickle", "wb")
    pickle.dump(predictor, file_to_store)
    file_to_store.close()



