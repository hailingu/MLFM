#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd


class NaiveBayesian:
    """naive bayesian model"""

    d1 = {}
    d2 = {}
    total = 0
    lmbd = 0

    def __init__(self, lmbd = 0):
        self.lmbd = lmbd

    def __call__(self, x):
        predict = {}
        prior = {}

        for j in self.d1.keys():
            prior[j] = float(self.d1[j] + self.lmbd) / (self.total + self.lmbd * x.shape[0])

            likelihood = 1
            for i in range(x.shape[0]):
                condition_key = "dim_" + str(i) + "|" + j
                if condition_key in self.d2.keys():
                    if x[i] in self.d2[condition_key].keys():
                        likelihood = likelihood * float(self.d2[condition_key][x[i]] + self.lmbd / self.d1[j] + self.lmbd * x.shape[0])
                    else:
                        likelihood = 0
                        break

            predict[j] = likelihood * prior[j]
        return max(predict, key=predict.get)

    def update(self, x, y, dim):
        self.total += 1

        if y in self.d1.keys():
            self.d1[y] = self.d1[y] + 1
        else:
            self.d1[y] = 1

        for i in range(dim):
            condition_key = "dim_" + str(i) + "|" + str(y)
            if condition_key in self.d2.keys():
                if x[i] in self.d2[condition_key].keys():
                    self.d2[condition_key][x[i]] = self.d2[condition_key][x[i]] + 1
                else:
                    self.d2[condition_key][x[i]] = 1
            else:
                self.d2[condition_key] = {}
                self.d2[condition_key][x[i]] = 1


def train(model, x, y):
    size = x.shape[0]
    dim = x.shape[1]
    for i in range(size):
        model.update(x[i], y[i], dim)


if __name__ == "__main__":
    data = (pd.read_csv('data/Iris.csv', sep=',', header=0)).values
    naive_bayesian_model = NaiveBayesian(0.01)
    train(naive_bayesian_model, data[:,1:-1], data[:, -1:].reshape(data.shape[0]))
    print(naive_bayesian_model(np.array([6.8,3.0,5.5,2.1])))