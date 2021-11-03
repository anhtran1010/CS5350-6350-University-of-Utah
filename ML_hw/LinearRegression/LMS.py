import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt

data_df = pd.read_csv("concrete/train.csv", header=None)
X_train = np.array(data_df.iloc[:,:-1].copy())
Y_train = np.array(data_df.iloc[:,-1].copy())

data_df_test = pd.read_csv("concrete/test.csv", header=None)
X_test = np.array(data_df_test.iloc[:,:-1].copy())
Y_test = np.array(data_df_test.iloc[:,-1].copy())
class LMS:
    def __init__(self):
        self.w =[]
        self.r = None
        self.step_costs = []

    def fit(self, X, Y, r, threshold=1e-6, SGD=False):
        self.r = r
        self.w = np.zeros(X[0].shape)
        converge = False
        step = 0
        step_costs = []
        while not converge :
            print("training step: ", step)
            cost = 0.5 * np.sum(np.square(Y - np.dot(X, self.w)))
            step_costs.append(cost)
            if SGD:
                old_w = self.w.copy()
                for x,y in zip(X,Y):
                    gradient = []
                    for j in range(len(self.w)):
                        error = y - np.dot(x, self.w)
                        gradient_j = error * x[j]
                        gradient.append(gradient_j)
                    self.w = self.w + np.multiply(r, gradient)
                weight_norm = LA.norm(self.w - old_w)
                if  weight_norm < threshold and step !=0:
                    converge = True
            else:
                gradient = []
                for j in range(len(self.w)):
                    gradient_j = 0
                    for x,y in zip(X,Y):
                        error = y-np.dot(x,self.w)
                        gradient_j += error*x[j]
                    gradient.append(-gradient_j)
                print(gradient)
                self.w -= np.multiply(r,gradient)
                print(LA.norm(-np.multiply(r,gradient)))
                if LA.norm(-np.multiply(r,gradient)) < threshold and step !=0:
                    converge = True
            step+=1
        self.step_costs = step_costs

    def predict(self, X, Y):
        return  0.5*np.sum(np.square(Y-np.dot(X,self.w)))

