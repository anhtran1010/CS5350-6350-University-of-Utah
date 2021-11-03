import numpy as np
import pandas as pd

def error_score(y_pred, y_target):
    return sum(np.not_equal(y_target, y_pred))/len(y_target)

class Perceptron:
    # constructor
    def __init__(self):
        self.w = None

    # model
    def forward(self, x):
        return 1 if (np.dot(self.w, x) >= 0) else 0

    # predictor to predict on the data based on w
    def predict(self, X):
        Y = []
        for x in X:
            result = self.forward(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, dataset, epochs=10, lr=1):
        dimension = dataset.iloc[:,:-1].shape[1]
        self.w = np.zeros(dimension+1)
        # for all epochs
        for i in range(epochs):
            shuffle_data = dataset.sample(frac=1)
            X = shuffle_data.iloc[:, :-1]
            X["bias"] = 1
            X = X.values
            Y = shuffle_data.iloc[:, -1].values
            for x, y in zip(X, Y):
                y_pred = self.forward(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x

class Voted_Perceptron:
    def __init__(self):
        self.w = None
        self.Cm_list = None
        self.wt_list = None
        self.bias_list = None

    # model
    def forward(self, x):
        pred = 0
        for i in range(len(self.wt_list)):
            sign = 1 if (np.dot(self.wt_list[i], x) >= 0) else -1
            pred += self.Cm_list[i]*sign
        return 1 if pred >=0 else 0

    # predictor to predict on the data based on w
    def predict(self, X):
        Y = []
        for x in X:
            result = self.forward(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, dataset, epochs=10, lr=1):
        dimension = dataset.iloc[:,:-1].shape[1]
        self.w = np.zeros(dimension+1)
        m = 0
        self.Cm_list = []
        self.wt_list = []
        # for all epochs
        for i in range(epochs):
            X = dataset.iloc[:, :-1]
            X["bias"] = 1
            X = X.values
            Y = dataset.iloc[:, -1].values
            Cm=1
            for x, y in zip(X, Y):
                y_pred = 1 if (np.dot(self.w, x) >= 0) else 0
                if y == 1 and y_pred == 0:
                    self.Cm_list.append(Cm)
                    self.w = self.w + lr * x
                    self.wt_list.append(self.w)
                    Cm = 1
                    m+=1
                elif y == 0 and y_pred == 1:
                    self.Cm_list.append(Cm)
                    self.w = self.w - lr * x
                    self.wt_list.append(self.w)
                    Cm = 1
                    m += 1
                else:
                    Cm+=1

class Avg_Perceptron:
    # constructor
    def __init__(self):
        self.w = None
        self.a = None

    # model
    def forward(self, x):
        return 1 if (np.dot(self.a, x) >= 0) else 0

    # predictor to predict on the data based on w
    def predict(self, X):
        Y = []
        for x in X:
            result = self.forward(x)
            Y.append(result)
        return np.array(Y)

    def fit(self, dataset, epochs=10, lr=1):
        dimension = dataset.iloc[:,:-1].shape[1]
        self.w = np.zeros(dimension+1)
        self.a = np.zeros(dimension+1)
        # for all epochs
        for i in range(epochs):
            X = dataset.iloc[:, :-1]
            X["bias"] = 1
            X = X.values
            Y = dataset.iloc[:, -1].values
            for x, y in zip(X, Y):
                y_pred = 1 if (np.dot(self.w, x) >= 0) else 0
                if y == 1 and y_pred == 0:
                    self.w = self.w + lr * x
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                self.a += self.w

raw_data_train = pd.read_csv("bank-note/bank-note/train.csv", header=None)
raw_data_test = pd.read_csv("bank-note/bank-note/test.csv", header=None)

X_test = raw_data_test.iloc[:, :-1]
X_test["bias"] = 1
X_test = X_test.values
Y_test = raw_data_test.iloc[:, -1].values

perceptron= Perceptron()
perceptron.fit(raw_data_train, 10, lr=1)
y_pred = perceptron.predict(X_test)
error = error_score(y_pred, Y_test)
print(error)
print(perceptron.w)

voted_perceptron = Voted_Perceptron()
voted_perceptron.fit(raw_data_train, 10, lr=1)
y_pred = voted_perceptron.predict(X_test)
error = error_score(y_pred, Y_test)
print(error)
for w, c in zip(voted_perceptron.wt_list, voted_perceptron.Cm_list):
    print("$",w,"$","&",c,"\\\\")


avg_perceptron = Avg_Perceptron()
avg_perceptron.fit(raw_data_train,10, lr=1)
y_pred = avg_perceptron.predict(X_test)
error = error_score(y_pred, Y_test)
print(error)
print(avg_perceptron.w)

