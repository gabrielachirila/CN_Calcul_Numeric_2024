import math

import numpy as np

epsilon = 10e-7


def F1(x, y):
    function = x ** 2 + y ** 2 - 2 * x - 4 * y - 1
    return function


def F2(x, y):
    function = 3 * x ** 2 - 12 * x + 2 * y ** 2 + 16 * y - 10
    return function


def F3(x, y):
    function = x ** 2 - 4 * x * y + 5 * y ** 2 - 4 * y + 3
    return function


def F4(x, y):
    function = x ** 2 * y - 2 * x * y ** 2 + 3 * x * y + 4
    return function


def gradient_F_analytic(F, x, y):
    if F == F1:
        grd = [2 * x - 2, 2 * y - 4]
    elif F == F2:
        grd = [6 * x - 12, 4 * y + 16]
    elif F == F3:
        grd = [2 * x - 4 * y, -4 * x + 10 * y - 4]
    elif F == F4:
        grd = [2 * x * y - 2 * y ** 2 + 3 * y, x ** 2 - 4 * x * y + 3 * x]
    return grd


def G1(F, x, y, h):
    sol = (3 * F(x, y) - 4 * F(x - h, y) + F(x - 2 * h, y)) / (2 * h)
    return sol


def G2(F, x, y, h):
    sol = (3 * F(x, y) - 4 * F(x, y - h) + F(x, y - 2 * h)) / (2 * h)
    return sol


def gradient_F(func, x, y, h=epsilon):
    sol = [G1(func, x, y, h), G2(func, x, y, h)]
    return sol


def calculate_learning_rate(F, x, y, gradient, beta):
    eta = 1
    p = 1
    while F(x - gradient[0], y - gradient[1]) > F(x, y) - eta/2 * np.linalg.norm(gradient) ** 2 and p < 8:
        eta *= beta
        p += 1
    return eta


def cossine_lr(curr_iteration=1, T_max=1000, init_lr=1e-1):
    lr = 1 / 2 * init_lr * (1 + math.cos(math.pi * curr_iteration / T_max))
    return lr


def descendent_gradient_method(F, h, k_max, beta, gradient_func):
    x = np.random.uniform(-10, 10)
    y = np.random.uniform(-10, 10)

    k = 0

    gradient = gradient_func(F, x, y)
    # learning_rate = calculate_learning_rate(F, x, y, gradient, beta)
    learning_rate = cossine_lr(k)
    x = x - learning_rate * gradient[0]
    y = y - learning_rate * gradient[1]
    k += 1

    while h <= learning_rate * np.linalg.norm(gradient) <= 10e10 and k <= k_max:
        gradient = gradient_func(F, x, y)
        # learning_rate = calculate_learning_rate(F, x, y, gradient, beta)
        learning_rate = cossine_lr(k)
        x -= learning_rate * gradient[0]
        y -= learning_rate * gradient[1]
        k += 1

    if learning_rate * np.linalg.norm(gradient) <= h:
        return x, y, k
    else:
        return None


def main():
    k_max = 30000
    beta = 0.8

    functions = [F1, F2, F3, F4]
    names = ["F1", "F2", "F3", "F4"]

    gradient_methods = [gradient_F, gradient_F_analytic]
    gradient_names = ["Approximation formula for gradient", "Analytic formula for gradient"]

    for i, F in enumerate(functions):
        print(names[i])
        for j, gradient_method in enumerate(gradient_methods):
            print(gradient_names[j])
            result = descendent_gradient_method(F, epsilon, k_max, beta, gradient_method)
            if result is not None:
                x, y, k = result
                print(f"x: {x}, y: {y}, iterations: {k}")
            else:
                print("No solution found")


main()
