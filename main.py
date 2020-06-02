import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sor_svr import SorSvr, linear_kernel_func, gauss_kernel_func, poly_kernel_func, gauss_kernel_func_n

x_train = np.linspace(-3, 3, 200)
x_test = np.linspace(-3, 3, 400)
func_name = "sinc(x)"


def func(x):
    return np.sinc(x)


# Hyper params for model data
epsilon = 0.1
sor_epsilon = 1e-3
omega = 1
C = 100
kernel = gauss_kernel_func
# Hyper params for noised data
noised_epsilon = 0.2
noised_sor_epsilon = sor_epsilon
noised_omega = 0.5
noised_C = 200
noised_kernel = gauss_kernel_func_n

noise_coeff = 0.1

# some magic
plot_data = False
errors = []
vectors_num = []


def plot_svr(svr, y_train, hash_train):
    if plot_data:
        plt.plot(x_train, y_train, "o", label="Тренировочная последовательность (l = {})".format(len(x_train)))
        plt.plot(x_train, func(x_train), "-", label="Истинная зависимость")
    # Train SVR
    svr.train(x_train, y_train)
    # Predict points
    y_test = [svr.predict(np.array([x])) for x in x_test]
    l = len(y_test)
    error = 0.0
    for i in range(0, l):
        error += np.abs(y_test[i] - func(x_test[i]))
    errors.append(error / l)
    if plot_data:
        plt.plot(x_test, y_test, "--", label="Восстановленная зависимость")
        error += func(x_test) - y_test
    # Draw support vectors
    x_supp = svr.get_support_vecs()
    vectors_num.append(len(x_supp))
    y_supp = [hash_train[x[0]] for x in x_supp]
    if plot_data:
        plt.plot(x_supp, y_supp, "r*", label="Опорные вектора (l = {})".format(len(x_supp)))
        plt.legend(loc=1)
        plt.show()


def plot_model_data(eps: float = epsilon):
    plt.figure()
    plt.title("{}. Модельные данные.".format(func_name))
    plt.xlabel("x")
    plt.ylabel("y")
    y_train = func(x_train)
    hash_train = {}  # for support vectors output
    for i in range(0, len(y_train)):
        hash_train[x_train[i]] = y_train[i]
    svr = SorSvr(eps, sor_epsilon, omega, C, kernel)
    plot_svr(svr, y_train, hash_train)


def plot_noised_data():
    plt.figure()
    plt.title("{}. Зашумленные данные.".format(func_name))
    plt.xlabel("x")
    plt.ylabel("y")
    y_train = func(x_train)
    hash_train = {}  # for support vectors output
    for i in range(0, len(y_train)):
        y_train[i] += np.random.uniform(-np.random.random() * noise_coeff, np.random.random() * noise_coeff)
        hash_train[x_train[i]] = y_train[i]
    svr = SorSvr(noised_epsilon, noised_sor_epsilon, noised_omega, noised_C, noised_kernel)
    plot_svr(svr, y_train, hash_train)


if __name__ == "__main__":
    eps_test = 0.15
    epss = []
    while eps_test >= 0.05:
        epss.append("{:1.2f}".format(eps_test))
        plot_model_data(eps_test)
        eps_test -= 0.01
    print(epss)
    print("ERRORS: {}".format(errors))
    print("SUPPORT VECTORS NUM: {}".format(vectors_num))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Изменение ошибки")
    plt.xlabel("epsilon")
    plt.ylabel("error")
    x = list(range(len(epss)))
    my_xticks = epss
    plt.xticks(x, my_xticks)
    plt.plot(x, errors)

    plt.subplot(1, 2, 2)
    plt.title("Изменение количества О.В.")
    plt.xlabel("epsilon")
    plt.ylabel("s.v. count")
    plt.xticks(x, my_xticks)
    plt.plot(x, vectors_num)
    plt.show()
    # plot_model_data()
    # plot_noised_data()
