from click import style
import numpy as np
from common import random_weights
import optim


class LinRegSGD:

    def __init__(self, dim, lr=1e-3) -> None:
        self.w = random_weights((dim+1,1))
        self.opt = optim.SGD(lr)

    def rmse(self):
        y_ = self.predict(self.X).reshape(-1)
        return np.sqrt(((self.y - y_)**2).mean())

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
        return X @ self.w

    def fit(self, X, y, n_iter):
        self.X, self.y = X, y
        hist = []
        X = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
        for i in range(n_iter):
            grad = X.T @ X @ self.w - 2 * X.T @ y.reshape(-1, 1)
            self.opt.step(zip([self.w], [grad]))
            hist.append(self.rmse())
        return hist


class LinRegExact:

    def __init__(self, dim) -> None:
        self.w = random_weights((dim+1,1))

    def rmse(self):
        y_ = self.predict(self.X).reshape(-1)
        return np.sqrt(((self.y - y_)**2).mean())

    def predict(self, X):
        X = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
        return X @ self.w

    def fit(self, X, y):
        self.X, self.y = X, y
        X_ = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
        self.w = np.linalg.inv(X_.T @ X_) @ X_.T @ y.reshape(-1, 1)
        return self.rmse()


if __name__ == "__main__":
    n, k = 100, 1
    mu, sigma = np.zeros(k+1), np.random.uniform(0.1, 1, (k+1,k+1))
    X = np.random.multivariate_normal(mu, sigma, n)
    X, y = X[:, :-1], X[:, -1]
    print(X.shape, y.shape)
    model = LinRegExact(k)
    print(model.fit(X, y), model.w)
    w0 = model.w.copy()
    model = LinRegSGD(k)
    print(model.fit(X, y, 1000)[-1], model.w)
    w1 = model.w

    import matplotlib.pyplot as plt
    xmin, xmax = X.min(), X.max()
    ymin, ymax = y.min(), y.max()
    plt.scatter(X, y)
    print(w0, w1)
    plt.plot([xmin, xmax], [xmin * w0[0] + w0[1], xmax*w0[0] + w0[1] ], c="red")
    plt.plot([xmin, xmax], [xmin * w1[0] + w1[1], xmax*w1[0] + w1[1] ], c="green")
    plt.legend(["", "exact", "sgd"])
    plt.show()