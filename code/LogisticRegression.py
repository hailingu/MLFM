#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def sigmoid(x, threshold = 0.5):
    return 1 if 1 / (1 + np.exp(-x)) > threshold else 0


class LogisticRegression:
    """Logistic Regression class"""
    w = None
    b = None

    def __init__(self):
        self.b = 0

    def __call__(self, x):
        if self.w is None:
            if isinstance(x, np.ndarray):
                self.w = np.zeros((x.shape[0]))
            else:
                self.w = np.zeros((1,))

        return sigmoid(np.dot(self.w, x) + self.b)

    def loss(self, y, yhat):
        return -1 * (np.dot(y,  np.log(yhat)) + np.dot((1-y),np.log(1-yhat)))

    def update(self, x, y, y_hat, eta):
        dw = -1 * np.dot(y - y_hat, x)
        db = -1 * np.sum(y - y_hat, 0)
        self.w = self.w - eta * dw
        self.b = self.b - eta * db


def train(model, x, y, eta = 0.1, epoch = 1):
    n = x.shape[0]
    y_hat = np.zeros((y.shape[0]))
    for i in range(epoch):
        loss = 0
        right = 0
        for j in range(n):
            y_hat[j] = model(x[j])
            loss += float(model.loss(y[j], y_hat[j]))

        model.update(x, y, y_hat, eta)
        for j in range(n):
            y_hat[j] = model(x[j])
            if y_hat[j] == y[j]:
                right += 1

        print("Epoch %d, Right Cnt is: %f" % (i, right))
    print("Training End.")


if __name__ == "__main__":
    data = np.array([[0.6863727, 0.17526787,  1.],
                     [0.80132839, 0.10523108, 1.],
                     [0.90775029, 0.14932357, 1.],
                     [0.80961084, 0.3694097,  1.],
                     [0.87508479, 0.60748736, 1.],
                     [0.70281723, 0.79587493, 1.],
                     [0.83433634, 0.15264152, 1.],
                     [0.93900273, 0.14149965, 1.],
                     [0.9024325,  0.5952603,  1.],
                     [0.85962927, 0.3794422,  1.],
                     [0.79602696, 0.99582496, 1.],
                     [0.91271228, 0.4730507,  1.],
                     [0.76601726, 0.5419725,  1.],
                     [0.75340131, 0.9640383,  1.],
                     [0.4891988,  0.51442063, 0.],
                     [0.17261154, 0.2748436,  0.],
                     [0.22090833, 0.86894599, 0.],
                     [0.07946182, 0.72304428, 0.],
                     [0.07177909, 0.93407657, 0.],
                     [0.27563872, 0.5806341,  0.],
                     [0.03817113, 0.79038495, 0.],
                     [0.23544452, 0.95657547, 0.],
                     [0.15161787, 0.34464048, 0.],
                     [0.30645413, 0.83393461, 0.],
                     [0.03985814, 0.28320299, 0.]])

    logistic_model = LogisticRegression()
    train(logistic_model, data[:, 0:2], data[:,2], epoch=10)