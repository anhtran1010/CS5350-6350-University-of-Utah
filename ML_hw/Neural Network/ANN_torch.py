import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class ANN(nn.Module):
    def __init__(self, weight_init_scheme=None, num_hidden_layers=3, num_hidden_size=5, act_func="RELU",
                 input_feature=32):
        super(ANN, self).__init__()
        self.sequential = nn.Sequential().to(dtype=torch.float32)
        if act_func == "RELU":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()
        for i in range(num_hidden_layers - 1):
            linear = nn.Linear(input_feature, num_hidden_size).to(dtype=torch.float32)
            if weight_init_scheme == "Xavier":
                nn.init.xavier_uniform_(linear.weight)
            else:
                nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
            self.sequential.add_module('layer_norm{}'.format(i), linear)
            if act_func == "RELU":
                self.sequential.add_module('act{}'.format(i), self.act)
            input_feature = num_hidden_size

        self.sequential.add_module('layer_norm{}'.format(num_hidden_layers - 1),
                                   nn.Linear(num_hidden_size, 1))

    def forward(self, input):
        output = self.sequential(input)
        x = torch.sigmoid(output)
        return x


def train(network,criterion, optimizer, X, y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.

    # Get the layer activations
    network.train()
    optimizer.zero_grad()
    output = network(X)
    # Compute the loss and the initial gradient
    y = torch.unsqueeze(y,1)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss

raw_data_train = pd.read_csv("./bank-note/bank-note/train.csv", header=None)
X_train = raw_data_train.iloc[:, :-1].values
Y_train = raw_data_train.iloc[:, -1].values

raw_data_test = pd.read_csv("./bank-note/bank-note/test.csv", header=None)
X_test = raw_data_test.iloc[:, :-1].values
Y_test = raw_data_test.iloc[:, -1].values
X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32)
Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32)
from tqdm import trange

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def predict(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    network.eval()
    sign = np.sign(network(X))
    return sign

from IPython.display import clear_output


#,10,25,50,100
num_hidden_layers = [3,5,9]
hidden_layer_sizes = [5,10,25,50,100]
learning_rate= 1e-5
alpha = 5
input_shape = X_train.shape[-1]

for num_hidden_layer in num_hidden_layers:
    for hidden_layer_size in hidden_layer_sizes:
        network = ANN(weight_init_scheme="Xavier",num_hidden_layers=num_hidden_layer, num_hidden_size=hidden_layer_size,act_func="Tanh",input_feature=input_shape)
        train_log = []
        test_log = []
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        for epoch in range(25):
            for x_batch, y_batch in iterate_minibatches(X_train, Y_train, batchsize=32, shuffle=True):
                 train(network, criterion, optimizer, x_batch, y_batch)

            network.eval()
            acc_train = np.mean((network(X_train).reshape(-1).detach().numpy().round() != Y_train.numpy()))
            train_log.append(acc_train)
            clear_output()
        acc_test = np.mean((network(X_test).reshape(-1).detach().numpy().round() != Y_test.numpy()))
        print("Number of Hidden Layer", num_hidden_layer)
        print("Hidden Layer Size", hidden_layer_size)
        print("Train error:", train_log[-1])
        print("Val error:", acc_test)

for num_hidden_layer in num_hidden_layers:
    for hidden_layer_size in hidden_layer_sizes:
        network = ANN(weight_init_scheme="He",num_hidden_layers=num_hidden_layer, num_hidden_size=hidden_layer_size,act_func="RELU",input_feature=input_shape)
        train_log = []
        test_log = []
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        for epoch in range(25):
            for x_batch, y_batch in iterate_minibatches(X_train, Y_train, batchsize=32, shuffle=True):
                 train(network, criterion, optimizer, x_batch, y_batch)

            network.eval()
            acc_train = np.mean((network(X_train).reshape(-1).detach().numpy().round() != Y_train.numpy()))
            train_log.append(acc_train)
            clear_output()
        acc_test = np.mean((network(X_test).reshape(-1).detach().numpy().round() != Y_test.numpy()))
        print("Number of Hidden Layer", num_hidden_layer)
        print("Hidden Layer Size", hidden_layer_size)
        print("Train error:", train_log[-1])
        print("Val error:", acc_test)