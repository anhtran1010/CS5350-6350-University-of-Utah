from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from Kaggle_data_preprocess import *

#
data_processor = DataPreprocess()
X_train, Y_train = data_processor.create_train_data_set("train_final.csv")
X_test = data_processor.create_test_data_set("test_final.csv")

bagging = BaggingClassifier(n_estimators=5000, max_samples=1500, max_features= 0.5, n_jobs = -1).fit(X_train, Y_train)
y_pred = bagging.predict(X_test)
data_frame = pd.DataFrame({"Prediction": y_pred})
id = data_frame.index + 1
data_frame.insert(0, "ID", id)
data_frame.to_csv("Kaggle_pred_bagged_5000_1500samples_0.5features", index=False)


# rtf = RandomForestClassifier(n_estimators=1000, max_features="log2", n_jobs = -1).fit(X_train, Y_train)
# y_pred = rtf.predict(X_test)
# data_frame = pd.DataFrame({"Prediction": y_pred})
# id = data_frame.index + 1
# data_frame.insert(0, "ID", id)
# data_frame.to_csv("Kaggle_pred_rtf_1000_log2", index=False)

#
# adaboost = AdaBoostClassifier(n_estimators=1000, learning_rate=0.5).fit(X_train, Y_train)
# y_pred = adaboost.predict(X_test)
# data_frame = pd.DataFrame({"Prediction": y_pred})
# id = data_frame.index + 1
# data_frame.insert(0, "ID", id)
# data_frame.to_csv("Kaggle_pred_adaboost_1000_0.5lr_no_missing_value", index=False)
#
# extra_tree = ExtraTreesClassifier(n_estimators=1000, max_features=0.5, n_jobs=-1).fit(X_train, Y_train)
# y_pred = extra_tree.predict(X_test)
# data_frame = pd.DataFrame({"Prediction": y_pred})
# id = data_frame.index + 1
# data_frame.insert(0, "ID", id)
# data_frame.to_csv("Kaggle_pred_extra_tree_1000_0.5", index=False)


