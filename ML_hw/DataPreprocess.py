import numpy as np
import pandas as pd

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

class DataPreprocess:
    def __init__(self, train_file, test_file, data_desc):
        self.train_file = train_file
        self.test_file = test_file
        self.data_desc = data_desc

    def create_data_set(self, is_train=True):
        description = open(self.data_desc, 'r')
        data_file = self.train_file if is_train else self.test_file
        raw_data = open(data_file,'r')
        lines = description.readlines()
        attributes = lines[16:36]
        attributes_value = []
        while (len(attributes) > 0):
            attribute = attributes.pop(0).strip()
            if is_integer(attribute[0:1]):
                if ' age ' in attribute:
                    attributes_value.append('age')
                elif 'education' in attribute:
                    attributes_value.append('education')
                else:
                    semi_ind = attribute.index(':')
                    attribute_value = attribute[4:semi_ind].strip()
                    attributes_value.append(attribute_value)

        data_dic = {}
        for attribute in attributes_value:
            data_dic[attribute] = []
        data_dic['label'] = []

        for line in raw_data:
            values = line.strip('\n').split(',')
            for index, key in enumerate(data_dic.keys()):
                data_dic[key].append(values[index])

        data_df = pd.DataFrame(columns=data_dic.keys())
        for key, value in data_dic.items():
            if 'unknown' in value:
                value_catergory = list(set(value))
                value_catergory = list(i for i in value_catergory if i != 'unknown')
                value = np.array(value)
                value_count = [(value == category).sum() for category in value_catergory]
                max_ind = value_count.index(max(value_count))
                replace_value = value_catergory[max_ind]
                value = [i if i != 'unknown' else replace_value for i in value]
            data_df[key] = value

        df_with_dummies = pd.get_dummies(data_df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan',
                                                           'contact', 'month', 'poutcome'], drop_first=False)

        S = np.array(df_with_dummies.drop('label', axis=1).copy())
        labels = np.array(df_with_dummies['label'].copy())
        labels_binary = []
        for label in labels:
            y = 1 if label == 'yes' else -1
            labels_binary.append(y)

        return S, labels_binary, attributes_value


