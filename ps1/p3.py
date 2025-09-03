import matplotlib.pyplot as plt
import numpy as np
import problem_set_1.src.util as util

from problem_set_1.src.linear_model import LinearModel

x_train, y_train = util.load_dataset("data/ds4_train.csv", add_intercept=True)
x_valid, y_valid = util.load_dataset("data/ds4_valid.csv", add_intercept=True)


class PoissonRegression(LinearModel):
    def h(self, theta, x):
        return np.exp(np.dot(x, theta))

    def fit(self, x, y):

        m, n = x.shape

        def next_step(theta):
            return self.step_size / m * x.T @ (y - self.h(theta, x))

        if self.theta is None:
            theta = np.zeros(n)
        else:
            theta = self.theta

        step = next_step(theta)
        while np.linalg.norm(step, 1) >= self.eps:
            theta += step
            step = next_step(theta)
        self.theta = theta

        self.theta = theta

    def predict(self, x):
        return np.exp(np.dot(x, self.theta))


clf = PoissonRegression(step_size=2e-7)
clf.fit(x_train, y_train)


def plot(y_label, y_pred, title):
    plt.plot(y_label, "go", label="label")
    plt.plot(y_pred, "rx", label="prediction")
    plt.suptitle(title, fontsize=12)
    plt.legend(loc="upper left")
    plt.show()


y_train_pred = clf.predict(x_train)
plot(y_train, y_train_pred, "Training Set")
y_valid_pred = clf.predict(x_valid)
plot(y_valid, y_valid_pred, "Validation Set")
