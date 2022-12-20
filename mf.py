import numpy as np
from torch import rand

# arg_min ||UV^T - R||^2 + alpha * (||U||**2 + ||V||**2)

def random_weights(shape):
    return np.random.normal(0,1,shape)

def _gradU(R, U, V, alpha):
    return 2 * (U @ V.T - R) @ V + alpha * U

def _gradV(R, U, V, alpha):
    return 2 * (V @ U.T - R.T) @ U + alpha * V


class Base:

    def __init__(self, n, m, k, alpha=0.5) -> None:
        self.U = random_weights((n, k))
        self.V = random_weights((m, k))
        self.alpha = alpha

    def _step(self):
        raise NotImplementedError()

    def rmse(self):
        return np.linalg.norm(self.R - self.U @ self.V.T)

    def fit(self, R, n_iter, supress=True):
        self.R = R
        hist = [self.rmse()]
        for i in range(n_iter):
            self._step()
            hist.append(self.rmse())
            if not supress:
                print(f"Iteration {i} rmse {hist[-1]:.4f}")
        return hist


class SGD(Base):

    def __init__(self, n, m, k, alpha=0.5, lr=1e-3) -> None:
        super().__init__(n, m, k, alpha)
        self.lr = lr

    def _step(self):
        U, V, R, alpha, lr = self.U, self.V, self.R, self.alpha, self.lr
        gU = _gradU(R, U, V, alpha)
        self.U[:] = U - lr * gU
        gV = _gradV(R, U, V, alpha)
        self.V[:] = V - lr * gV


class ALS(Base):

    def __init__(self, n, m, k, alpha=0.5) -> None:
        super().__init__(n, m, k, alpha)

    def _step(self):
        U, V, R, alpha = self.U, self.V, self.R, self.alpha
        self.U[:] = 2 * R @ V @ np.linalg.inv(2 * V.T @ V + alpha * np.eye(k))
        self.V[:] = 2 * R.T @ U @ np.linalg.inv(2 * U.T @ U + alpha * np.eye(k))


if __name__ == '__main__':
    m, n, k = 500, 600, 20
    R = np.random.normal(0, 1, (n, m))
    model = ALS(n, m, k)
    print(np.linalg.norm(R - model.U @ model.V.T))
    model.fit(R, 100, False)