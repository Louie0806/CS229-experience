import math
import matplotlib.pyplot as plt
import numpy as np
import problem_set_1.src.util as util

from problem_set_1.src.linear_model import LinearModel

x_train, y_train = util.load_dataset("data/ds5_train.csv", add_intercept=True)
x_valid, y_valid = util.load_dataset("data/ds5_valid.csv", add_intercept=True)
x_test, y_test = util.load_dataset("data/ds5_test.csv", add_intercept=True)
plt.xlabel("x_1")
plt.ylabel("y")
plt.plot(x_train[:, -1], y_train, "bx", linewidth=2)


class LocallyWeightedLinearRegression(LinearModel):
    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x):

        l, n = x.shape

        # Reshape the input x by adding an additional dimension so that it can broadcast
        w_vector = np.exp(
            -np.linalg.norm(self.x - np.reshape(x, (l, -1, n)), ord=2, axis=2) ** 2
            / (2 * self.tau**2)
        )

        # Turn the weights into diagonal matrices, each corresponds to a single input. Shape (l, m, m)
        w = np.apply_along_axis(np.diag, axis=1, arr=w_vector)

        # Compute theta for each input x^(i). Shape (l, n)
        theta = np.linalg.inv(self.x.T @ w @ self.x) @ self.x.T @ w @ self.y

        return np.einsum("ij,ij->i", x, theta)


clf = LocallyWeightedLinearRegression(tau=0.5)
clf.fit(x_train, y_train)


def plot(x, y_label, y_pred, title):
    plt.figure()
    plt.plot(x[:, -1], y_label, "bx", label="label")
    plt.plot(x[:, -1], y_pred, "ro", label="prediction")
    plt.suptitle(title, fontsize=12)
    plt.legend(loc="upper left")
    plt.show()


y_train_pred = clf.predict(x_train)
plot(x_train, y_train, y_train_pred, "Training Set")

y_valid_pred = clf.predict(x_valid)
plot(x_valid, y_valid, y_valid_pred, "Validation Set")


# 调tau,要求最小MSE
taus = [3e-2, 5e-2, 1e-1, 5e-1, 1e0, 1e1]
lowest_mse = math.inf
best_tau = taus[0]

for tau in taus:
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train)
    y_valid_pred = clf.predict(x_valid)

    mse = np.mean((y_valid_pred - y_valid) ** 2)
    if mse < lowest_mse:
        lowest_mse = mse
        best_tau = tau

    plot(x_valid, y_valid, y_valid_pred, f"Validation Set (Tau = {tau}, MSE = {mse})")

print(f"Tau = {best_tau} achieves the lowest MSE on the validation set.")
