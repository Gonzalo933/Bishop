import numpy as np
import matplotlib.pyplot as plt


np.random.seed(8)


class PolynomialModel:
    def __init__(self, M):
        self.M = M
        self.coeffs = np.ones([M, 1])

    def y(self, x):
        # calculating y = X @ w
        vandermonde_matrix = x ** np.arange(self.M)
        # x**powers  creates a Vandermonde Matrix
        return vandermonde_matrix @ self.coeffs

    def train(self, x_train, y_train):
        vander = x_train ** np.arange(self.M)
        self.coeffs = np.linalg.inv(vander.T @ vander) @ vander.T @ y_train
        return self


def f(x):
    return np.sin(2 * np.pi * x)


N = 100
N_train = int(0.1 * N)
noise_std = 0.1
x = np.linspace(0, 1, num=N)[:, None]
x_train = np.linspace(0, 1, num=N_train)[:, None]
y_train = f(x_train) + np.random.normal(scale=noise_std, size=N_train)[:, None]
model = PolynomialModel(3)

plt.figure()
plt.xlabel("x")
plt.ylabel("y")

# Plot function from which data was generated
plt.plot(x, f(x))
# Plot training data
plt.plot(x_train, y_train, "o")
# Plot untrained model
plt.plot(x_train, model.y(x_train))
# Plot trained model
model.train(x_train, y_train)
plt.plot(x_train, model.y(x_train))
plt.show()
