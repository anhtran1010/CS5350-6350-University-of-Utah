from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class DataPreprocess:
    def __init__(self):
        self.data_set = None

    def create_train_data_set(self, data_file):
        raw_data = pd.read_csv(data_file)
        categorical_data = ["workclass", "marital.status", "occupation", "relationship", "race", "sex", "native.country", "education"]
        for (columnName, columnData) in raw_data.iteritems():
            if columnName in categorical_data:
                column = raw_data[columnName].to_numpy()
                most_frequent = Counter(column).most_common(1)[0][0]
                raw_data[columnName].str.replace("?", most_frequent)
            else:
                column = raw_data[columnName].to_numpy()
                most_frequent = np.bincount(column).argmax()
                raw_data[columnName].replace("?", most_frequent)

        df_with_dummies = pd.get_dummies(raw_data, columns=['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race',
                                                           'sex', 'native.country'], drop_first=False)
        num_samples = len(np.array(df_with_dummies['income>50K'].copy()))
        holand = np.zeros(num_samples)
        df_with_dummies.insert(82, "native.country_Holand-Netherlands", holand, True)
        features = np.array(df_with_dummies.drop('income>50K', axis=1).copy())
        labels = np.array(df_with_dummies['income>50K'].copy())
        # st_x = StandardScaler()
        # features = st_x.fit_transform(features)
        return features, labels

    def create_test_data_set(self, data_file):
        raw_data = pd.read_csv(data_file)
        categorical_data = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex",
                            "native.country"]
        for (columnName, columnData) in raw_data.iteritems():
            if columnName in categorical_data:
            # if columnName =="workclass" or "education" or "marital.status" or "occupation" or "relationship" or "race" or "sex" or "native.country":
                column = raw_data[columnName].to_numpy()
                most_frequent = Counter(column).most_common(1)[0][0]
                raw_data[columnName].str.replace("?", most_frequent)
            else:
                column = raw_data[columnName].to_numpy()
                most_frequent = np.bincount(column).argmax()
                raw_data[columnName].replace("?", most_frequent)
        df_with_dummies = pd.get_dummies(raw_data, columns=['workclass', 'education', 'marital.status', 'occupation',
                                                            'relationship', 'race',
                                                            'sex', 'native.country'], drop_first=False)
        features = np.array(df_with_dummies.drop('ID', axis=1).copy())
        # st_x = StandardScaler()
        # features = st_x.fit_transform(features)
        return features

data_processor = DataPreprocess()
X_train, Y_train = data_processor.create_train_data_set("train_final.csv")
X_test = data_processor.create_test_data_set("test_final.csv")
import pickle as pkl
pkl.dump((X_train, Y_train), open("train_set_kaggle.pkl", "wb"))
pkl.dump((X_test), open("test_set_kaggle.pkl", "wb"))