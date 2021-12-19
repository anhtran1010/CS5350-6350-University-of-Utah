import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from tqdm import tqdm_notebook, trange
from sklearn.metrics import accuracy_score
import numpy as np

class NeuralNet():

    def __init__(self, architecture):
        self.architecture = architecture
        self.params = self._initialize_params(architecture)

    def _initialize_params(self, architecture):
        params = {}
        for id_, layer in enumerate(architecture):
            layer_id = id_ + 1

            input_dim = layer['input_dim']
            output_dim = layer['output_dim']

            params['W' + str(layer_id)] = np.random.rand(output_dim,input_dim)*0.1
            params['b' + str(layer_id)] = np.zeros((output_dim, 1))

        return params

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid_backward(self, dA, z_curr):
        sig = self.sigmoid(z_curr)
        return sig * (1 - sig) * dA

    def relu_backward(self, dA, z_curr):
        dz = np.array(dA, copy=True)
        dz[z_curr <= 0] = 0
        return dz

    def _forward_prop_this_layer(self, A_prev, W_curr, b_curr, activation_function):
        z_curr = np.dot(W_curr, A_prev) + b_curr

        if activation_function is 'relu':
            activation = self.relu
        elif activation_function is 'sigmoid':
            activation = self.sigmoid
        else:
            raise Exception(f"{activation_function} is not supported, Only sigmoid, relu are supported")

        return activation(z_curr), z_curr

    def _forward(self, X):
        cache = {}
        A_current = X
        for layer_id_prev, layer in enumerate(self.architecture):
            current_layer_id = layer_id_prev + 1

            A_previous = A_current
            activation = layer['activation']

            W_curr = self.params['W' + str(current_layer_id)]
            b_curr = self.params['b' + str(current_layer_id)]

            A_current, Z_curr = self._forward_prop_this_layer(A_previous, W_curr,
                                                              b_curr, activation)

            cache['A' + str(layer_id_prev)] = A_previous
            cache['Z' + str(current_layer_id)] = Z_curr

        return A_current, cache

    def _criterion(self, y, yhat):
        m = yhat.shape[1]
        cost = -1 / m * (np.dot(y, np.log(yhat).T) + np.dot(1 - y, np.log(1 - yhat).T))
        return np.squeeze(cost)

    def _backprop_this_layer(self, da_curr, z_curr, W_curr, b_curr, A_prev, activation_function):
        if activation_function is 'sigmoid':
            activation_back = self.sigmoid_backward
        elif activation_function is 'relu':
            activation_back = self.relu_backward
        else:
            raise Exception('need sigmoid or relu')
        m = A_prev.shape[1]

        dz_curr = activation_back(da_curr, z_curr)
        dw_curr = np.dot(dz_curr, A_prev.T) / m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m
        da_prev = np.dot(W_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr

    def _backward(self, ytrue, ypred, cache):
        grads = {}
        m = ytrue.shape[1]
        da_prev = np.divide(1 - ytrue, 1 - ypred) - np.divide(ytrue, ypred)

        for prev_layer_id, layer in reversed(list(enumerate(self.architecture))):
            layer_id = prev_layer_id + 1
            activation = layer['activation']

            da_curr = da_prev

            A_prev = cache['A' + str(prev_layer_id)]
            Z_curr = cache['Z' + str(layer_id)]

            W_curr = self.params['W' + str(layer_id)]
            b_curr = self.params['b' + str(layer_id)]

            da_prev, dw_curr, db_curr = self._backprop_this_layer(
                da_curr, Z_curr, W_curr, b_curr, A_prev, activation)

            grads["dw" + str(layer_id)] = dw_curr
            grads['db' + str(layer_id)] = db_curr

        return grads

    def update(self, grads, learning_rate):
        for layer_id, layer in enumerate(self.architecture, 1):
            self.params['W' + str(layer_id)] -= learning_rate * grads['dw' + str(layer_id)]
            self.params['b' + str(layer_id)] -= learning_rate * grads['db' + str(layer_id)]

    def fit(self, X, y, epochs, learning_rate, verbose=True, show_loss=True):
        loss_history, error_history = [], []
        for epoch in tqdm_notebook(range(epochs), total=epochs, unit='epoch'):
            indices = np.random.permutation(len(X))
            X,y = X[indices], y[indices]
            X_train, y_train = X.T, y.reshape((y.shape[0], -1)).T
            lr = learning_rate / (1 + learning_rate * epoch / alpha)
            yhat, cache = self._forward(X_train)
            loss = self._criterion(y, yhat)
            loss_history.append(loss)
            yacc = yhat.copy()
            yacc[yacc > 0.5] = 1
            yacc[yacc <= 0.5] = 0

            error = np.sum(y_train[0] != yacc[0]) / (yacc.shape[1])
            error_history.append(error)

            grads_values = self._backward(y_train, yhat, cache)

            self.update(grads_values, lr)

            if (epoch % 100 == 0):
                if (verbose):
                    print("Epoch: {:05} - cost: {:.5f} - error: {:.5f}".format(epoch, loss, error))

        fig = plt.figure(figsize=(12, 10))
        plt.plot(range(epochs), loss_history, 'r-')
        plt.plot(range(epochs), error_history, 'b--')
        plt.legend(['Training_loss', 'Training_Error'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.show()

    def predict(self, X):
        yhat, _ = self._forward(X)
        yhat[yhat > 0.5] = 1
        yhat[yhat <= 0.5] = 0
        return np.squeeze(yhat)

raw_data_train = pd.read_csv("./bank-note/bank-note/train.csv", header=None)
X_train = raw_data_train.iloc[:, :-1].values
Y_train = raw_data_train.iloc[:, -1].values

raw_data_test = pd.read_csv("./bank-note/bank-note/test.csv", header=None)
X_test = raw_data_test.iloc[:, :-1].values
Y_test = raw_data_test.iloc[:, -1].values

# #,10,25,50,100
hidden_layer_sizes = [5,10,25,50]
learning_rate= 0.01
alpha = 5
input_shape = X_train.shape[-1]

for hidden_layer_size in hidden_layer_sizes:
    NN_ARCHITECTURE = [
        {"input_dim": input_shape, "output_dim": hidden_layer_size, "activation": "relu"},  # Input Layer
        {"input_dim": hidden_layer_size, "output_dim": hidden_layer_size, "activation": "relu"},  # Hidden Layer -- 1
        {"input_dim": hidden_layer_size, "output_dim": hidden_layer_size, "activation": "relu"},  # Second Hidden Layer
        {"input_dim": hidden_layer_size, "output_dim": 1, "activation": "sigmoid"},  # Output Layer
    ]
    net = NeuralNet(NN_ARCHITECTURE)
    net.fit(X_train, Y_train, epochs=500, learning_rate=learning_rate, verbose=True, show_loss=True)
    predictions = net.predict(X_test.T)
    score = 1 - accuracy_score(Y_test, predictions)
    print(score)
