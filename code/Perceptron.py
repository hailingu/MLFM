#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math


class Perceptron(object):
    w = None
    b = None
    eta = None
    dw = None
    db = None

    def __init__(self, n, eta):
        self.w = np.random.uniform(0,1,(n,))
        self.dw = np.zeros((n,))
        self.b = 0
        self.eta = eta

    def __call__(self, x):
        w_norm_2_sqrt = math.sqrt((np.dot(self.w, self.w)))
        return (np.dot(self.w, x) + self.b) # / w_norm_2_sqrt

    def update(self, x, y):
        w_norm_2 = np.dot(self.w, self.w)
        w_norm_2_sqrt = math.sqrt(w_norm_2)
        n = self.dw.shape[0]

        for i in range(n):
            # self.dw[i] = y * x[i] / w_norm_2_sqrt - self.w[i] * (np.dot(self.w, x) + self.b) / (w_norm_2_sqrt * w_norm_2_sqrt * w_norm_2_sqrt)
            self.dw[i] = y * x[i]

        self.db = y # / w_norm_2_sqrt
        self.w = self.w - self.eta * self.dw
        self.b = self.b - self.eta * self.db

    def loss(self, x, y):
        return -1 * y * self(x)

    def plot(self, data, index):
        r = data[(data[:, :] > 0)[:, -1]]
        b = data[(data[:, :] == 0)[:, -1]]
        f = plt.figure(index)
        yl = np.zeros((100, ))
        y2 = np.zeros((100,))
        xl = np.random.uniform(-1,1,(100,))

        skweights = np.array([2.90162864, -1.12481468])
        skinceptr = -1
        for i in range(100):
            yl[i] = (np.dot(self.w[0:-1], xl[i]) + self.b) / (-1 * self.w[-1])
            y2[i] = (np.dot(skweights[0], xl[i]) + skinceptr) / (-1 * skweights[-1])

        plt.scatter(r[:, 0], r[:, 1], marker='o', color='r', linewidths=0.5)
        plt.scatter(b[:, 0], b[:, 1], marker='x', color='b', linewidths=0.5)
        plt.plot(xl, yl, color="gray", linewidth = 1)
        plt.plot(xl, y2, color = "green", linewidth = 1)
        plt.axis([-1,1,-1,1])
        f.savefig("figure_" + str(index) + ".png")
        plt.close(index)
        print("index: %d, self.w: %f, %f, self.b: %f" % (index, self.w[0], self.w[1], self.b))


def train(percetron, data, epoch = 1):
    n = data.shape[0]
    right = 0

    for i in range(epoch):
        if i % 1 == 0:
            percetron.plot(data, i)
            print("Epoch: %d, self.w: %f, %f, self.b: %f, right: %d" %
                  (i, perceptron.w[0], perceptron.w[1], perceptron.b, right))

        if right == n:
            break

        right = 0
        for j in range(n):
            y = data[j][2]
            if y == 0.0:
               y = -1

            loss = percetron.loss(data[j][0:2], y)

            if loss > 0:
                percetron.update(data[j][0:2], -1 * y)
            else:
                right += 1


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

    perceptron = Perceptron(2, 0.1)
    train(perceptron, data, 30)
