import numpy as np
import pandas as pd


def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))


def logistic_regression(X, Y, learning_rate=0.01, n_epoch=100, variance=1, alpha=5, MAP=True):
    weight = np.random.normal(loc=0.0, scale=variance, size=X.shape[1])

    for epoch in range(n_epoch):
        lr = learning_rate / (1 + learning_rate * epoch / alpha)
        count = 0
        for x, y in zip(X, Y):
            scores = np.dot(x, weight)
            prediction = sigmoid(scores)
            error = y * prediction
            if MAP:
                gradient = (1 - sigmoid(error)) * y * x + 2 * weight / variance
                print(gradient)
            else:
                gradient = (1 - sigmoid(error)) * y * x
            weight -= lr * gradient
            count+=1
    return weight

raw_data_train = pd.read_csv("./bank-note/train.csv", header=None)
X_train = raw_data_train.iloc[:, :-1].values
Y_train = raw_data_train.iloc[:, -1].values
Y_train = np.array([1 if i ==1 else -1 for i in Y_train])

raw_data_test = pd.read_csv("./bank-note/test.csv", header=None)
X_test = raw_data_test.iloc[:, :-1].values
Y_test = raw_data_test.iloc[:, -1].values
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
print("With MAP")
for variance in variances:
    weights = logistic_regression(X_train, Y_train, variance=variance)
    weights = np.expand_dims(weights, 1)
    train_error = np.mean(np.sign(np.dot(X_train,weights)) != Y_train)
    print(train_error)
    test_error = np.mean(np.sign(np.dot(X_test, weights)) != Y_test)
    print(test_error)
print("-------------------")

print("Without MAP")
for variance in variances:
    weights = logistic_regression(X_train, Y_train, variance=variance, MAP=False)
    weights = np.expand_dims(weights, 1)
    train_error = np.mean(np.sign(np.dot(X_train,weights)) != Y_train)
    print(train_error)
    test_error = np.mean(np.sign(np.dot(X_test, weights)) != Y_test)
    print(test_error)

