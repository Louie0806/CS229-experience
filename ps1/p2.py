import numpy as np
import problem_set_1.src.util as util

from problem_set_1.src.linear_model import LinearModel

x_train, y_train = util.load_dataset("data/ds3_train.csv", add_intercept=True)
_, t_train = util.load_dataset("data/ds3_train.csv", label_col="t")
x_valid, y_valid = util.load_dataset("data/ds3_valid.csv", add_intercept=True)
_, t_valid = util.load_dataset("data/ds3_valid.csv", label_col="t")
x_test, y_test = util.load_dataset("data/ds3_test.csv", add_intercept=True)
_, t_test = util.load_dataset("data/ds3_test.csv", label_col="t")


# 逻辑回归
class LogisticRegression(LinearModel):
    def fit(self, x, y):
        def h(theta, x):
            return 1 / (1 + np.exp(-np.dot(x, theta)))

        def gradient(theta, x, y):
            m, _ = x.shape
            return -1 / m * np.dot(x.T, y - h(theta, x))

        def hessian(theta, x):
            m, _ = x.shape
            h_theta_x = np.reshape(h(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, h_theta_x * (1 - h_theta_x) * x)

        def next_theta(theta, x, y):
            return theta - np.dot(
                np.linalg.inv(hessian(theta, x)), gradient(theta, x, y)
            )

        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)
        old_theta = self.theta
        new_theta = next_theta(old_theta, x, y)
        while np.linalg.norm(old_theta - new_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)
        self.theta = new_theta

    def predict(self, x):
        return x @ self.theta >= 0


log_reg = LogisticRegression()
log_reg.fit(x_train, t_train)
util.plot(x_train, t_train, log_reg.theta)
print("Theta is: ", log_reg.theta)
print("The accuracy on training set is: ", np.mean(t_train == log_reg.predict(x_train)))


util.plot(x_test, t_test, log_reg.theta)
print("Theta is: ", log_reg.theta)
print("The accuracy on test set is: ", np.mean(t_test == log_reg.predict(x_test)))
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
util.plot(x_test, y_test, log_reg.theta)


def h(theta, x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


v_plus = x_valid[y_valid == 1]
alpha = h(log_reg.theta, v_plus).mean()  # 计算alpha


def predict(theta, x):
    return h(theta, x) / alpha >= 0.5  # 边界条件


theta_prime = log_reg.theta + np.log(2 / alpha - 1) * np.array(
    [1, 0, 0]
)  # theta进行修正

util.plot(x_test, y_test, theta_prime)
print("Theta_prime is: ", theta_prime)
print(
    "The accuracy on test set is: ", np.mean(predict(log_reg.theta, x_test) == t_test)
)
