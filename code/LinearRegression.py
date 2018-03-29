import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return 3*x + 2

noise = np.random.uniform(-1, 1, 100)
X = np.random.uniform(0, 1, 100)
Y = f(X) + noise

class SimpleLinearRegression:
    """Simple Linear Model"""
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

        return np.dot(self.w, x) + self.b

    def loss(self, y, yhat):
        return math.pow(y - yhat, 2)

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
        for j in range(n):
            y_hat[j] = model(x[j])
            loss += float(model.loss(y[j], y_hat[j]))

        model.update(x, y, y_hat, eta)
        print("Epoch %d, loss is: %f" % (i, loss / n))
    print("Training End.")


if __name__ == "__main__":
    simplelinearmodel = SimpleLinearRegression()
    train(simplelinearmodel, X, Y, 0.01, 1000)

    plt.scatter(X, Y, color="red", linewidth=1)

    yl = np.zeros((100,))
    for i in range(100):
        yl[i] = simplelinearmodel(X[i])

    plt.plot(X, yl, color="gray", linewidth=1)
    plt.savefig("f2.1.png")
    plt.show()

    print(simplelinearmodel.w, simplelinearmodel.b)