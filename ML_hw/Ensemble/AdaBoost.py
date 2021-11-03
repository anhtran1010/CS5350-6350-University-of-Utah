from DecisionTreeNumeric import *
import numpy as np
import pandas as pd

class AdaBoost:
    def __init__(self):
        self.alphas = []
        self.H = []
        self.T = None
        self.training_errors = []
        self.prediction_errors = []

    def fit(self, X, attributes, y, T=100):
        self.alphas = []
        self.training_errors = []
        self.H = []
        self.T = T
        labels = y
        d_i = np.ones(len(y)) * 1 / len(y)
        for t in range(0, self.T):
            h = DecisionTreeClassifier(S=X, attributes=attributes, labels=y, sample_weight= d_i, max_depth=1)
            h.id3()
            y_pred = h.predict(X)
            self.H.append(h)

            error_t = (sum(d_i * (np.not_equal(y, y_pred)).astype(int))) / sum(d_i)
            self.training_errors.append(error_t)

            alpha_t = np.log((1 - error_t) / error_t)
            self.alphas.append(alpha_t)

            d_i = d_i * np.exp(alpha_t * (np.not_equal(labels, y_pred)).astype(int))

        assert len(self.H) == len(self.alphas)

    def predict(self, X):
        weak_preds = pd.DataFrame(index=range(len(X)), columns=range(self.T))

        for t in range(self.T):
            y_pred_t = self.H[t].predict(X) * self.alphas[t]
            if 1.0 in y_pred_t:
                print(True)
            weak_preds.iloc[:, t] = y_pred_t

        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        return y_pred

