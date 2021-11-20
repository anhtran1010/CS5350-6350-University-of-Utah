import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
from sklearn.utils import shuffle
from cvxopt import matrix, solvers

C_list = [100/873, 500/873, 700/873]
learning_rate = 1e-3

def compute_cost(W, X, Y):
    C = 100/873
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances = np.maximum(0, distances)  # equivalent to max(0, distance)
    hinge_loss = C * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    C = 100 / 873
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])
    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    if max(0, distance) == 0:
        di = W
    else:
        di = W - (C * Y_batch * X_batch)
        dw += di
    dw = dw  # average
    return dw


def sgd(features, outputs, alpha=5, lr_with_alpha = True):
    features.insert(loc=len(features.columns), column='bias', value=1)
    features = features.values
    T = 100
    weights = np.zeros(features.shape[1])
    # stochastic gradient descent
    for t in range(1, T):
        if lr_with_alpha:
            learning_rate_t = learning_rate / (1 + learning_rate*t/alpha)
        else:
            learning_rate_t = learning_rate / (1 + t)
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate_t * ascent)
    return weights

def gaussian_kernel(x, z, sigma):
    n = x.shape[0]
    m = z.shape[0]
    xx = np.dot(np.sum(np.power(x, 2), 1).reshape(n, 1), np.ones((1, m)))
    zz = np.dot(np.sum(np.power(z, 2), 1).reshape(m, 1), np.ones((1, n)))
    return np.exp(-(xx + zz.T - 2 * np.dot(x, z.T)) / sigma)


def linear_kernel(x, z):
    return np.matmul(x, z.T)

def dual_sgd(features, outputs, C = 100/873, gamma = 0.1, kernel_scheme='linear'):
    m, n = features.shape
    K = gaussian_kernel(features, features, gamma)
    P = matrix(np.matmul(outputs, outputs.T) * K)
    q = matrix(np.ones((m, 1)) * -1)
    outputs = outputs.astype('double')
    A = matrix(outputs.reshape(1, -1))
    b = matrix(np.zeros(1))
    G = matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)))

    optimizer_func = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(optimizer_func['x'])
    ind = (alphas > 1e-4).flatten()
    sv = features[ind]
    sv_y = outputs[ind]
    alphas = alphas[ind]
    b = sv_y - np.sum(linear_kernel(sv, sv) * alphas * sv_y, axis=0)
    b = np.sum(b) / b.size
    w = np.sum(alphas * sv_y[:,None] * sv, axis=0)
    prod = np.sum(sv_y[:, None] * (gaussian_kernel(sv, features, gamma) * alphas), axis=0) + b
    dual_sgd_pred = np.sign(prod)
    error_train = sum(np.not_equal(outputs, dual_sgd_pred)) / len(outputs)
    print("training error: ", error_train)
    return w ,b, sv, sv_y, alphas

def count_same(array_1, array_2):
    same = 0
    for array in array_1:
        if array in array_2:
            same+=1
    return same

raw_data_train = pd.read_csv("./bank-note/bank-note/train.csv", header=None)
X_train = raw_data_train.iloc[:, :-1]
# X_train.insert(loc=len(X_train.columns), column='bias', value=1)
# X_train_with_bias = X_train.values
# X_train_no_bias = X_train.iloc[:, :-1]
Y_train = raw_data_train.iloc[:, -1].values
Y_train = np.array([1 if i ==1 else -1 for i in Y_train])
W_alpha = sgd(X_train.copy(), Y_train)
W = sgd(X_train.copy(), Y_train, lr_with_alpha=False)
print(W)
W_diff = W_alpha - W
print("weights without alpha for lr: ", W)
print("weights with alpha for lr: ", W_alpha)
print("Weight difference: ", W_diff)
y_pred_alpha = np.array([])
y_pred = np.array([])
for i in range(X_train.shape[0]):
    X_train_with_bias = X_train.copy()
    X_train_with_bias.insert(loc=len(X_train.columns), column='bias', value=1)
    X_train_with_bias = X_train_with_bias.values
    yp = np.sign(np.dot(X_train_with_bias[i], W))
    y_pred = np.append(y_pred, yp)

    yp_alpha = np.sign(np.dot(X_train_with_bias[i], W_alpha))
    y_pred_alpha = np.append(y_pred_alpha, yp_alpha)
error = sum(np.not_equal(Y_train, y_pred)) / len(Y_train)
error_alpha = sum(np.not_equal(Y_train, y_pred_alpha)) / len(Y_train)
print("Training Error for lr without alpha: ", error)
print("Training Error for lr with alpha: ", error_alpha)
print("Training error diff: ", error - error_alpha)
raw_data_test = pd.read_csv("./bank-note/bank-note/test.csv", header=None)

X_test = raw_data_test.iloc[:, :-1]
X_test.insert(loc=len(X_test.columns), column='bias', value=1)
X_test_with_bias = X_test.values
X_test_no_bias = X_test.iloc[:,:-1].values
Y_test = raw_data_test.iloc[:, -1].values
Y_test = np.array([1 if i ==1 else -1 for i in Y_test])
y_pred_alpha_test = np.array([])
y_pred_test = np.array([])
for i in range(X_test.shape[0]):
    ypt = np.sign(np.dot(X_test_with_bias[i], W))
    y_pred_test = np.append(y_pred_test, ypt)

    ypt_alpha = np.sign(np.dot(X_test_with_bias[i], W_alpha))
    y_pred_alpha_test = np.append(y_pred_alpha_test, ypt_alpha)
error = sum(np.not_equal(Y_test, y_pred_test)) / len(Y_test)
error_alpha = sum(np.not_equal(Y_test, y_pred_alpha_test)) / len(Y_test)
print("Testing Error for lr without alpha: ", error)
print("Testing Error for lr with alpha: ", error_alpha)
print("Testing error diff: ", error - error_alpha)
#

_, _, sv, _, _ = dual_sgd(X_train.to_numpy(), Y_train, 500/873, 0.1)
previous_sv = sv
gamma_list = [0.5, 1.0, 5.0, 100.0]
for C in C_list:
    for gamma in gamma_list:
        print("C: ", C)
        print("gamma: ", gamma)
        w, b, sv, sv_y, alphas = dual_sgd(X_train.to_numpy(), Y_train, C, gamma)
        print(count_same(sv, previous_sv))
        previous_sv = sv

        prod_test = np.sum(sv_y[:,None]* (gaussian_kernel(sv, X_test_no_bias, gamma) * alphas), axis=0) + b
        dual_sgd_pred_test = np.sign(prod_test)
        error_test= sum(np.not_equal(Y_test, dual_sgd_pred_test)) / len(Y_test)
        print("error for testing: ", error_test)

