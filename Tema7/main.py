import numpy as np
import random



def horner(coefficients, v):
    result = 0
    for coefficient in coefficients:
        result = result * v + coefficient
    return result

def calculate_R(coefficients):
    A = max(abs(coef) for coef in coefficients)
    return (abs(coefficients[0]) + A )/ abs(coefficients[0])


def muller_method(coefficients, x0, x1, x2, epsilon, max_iterations):
    k = 3
    roots = []

    h0 = x1 - x0
    h1 = x2 - x1
    f0 = horner(coefficients, x0)
    f1 = horner(coefficients, x1)
    f2 = horner(coefficients, x2)

    delta0 = (f1 - f0) / h0
    delta1 = (f2 - f1) / h1

    a = (delta1 - delta0) / (h1 + h0)
    b = a * h1 + delta1
    c = f2

    if b ** 2 - 4 * a * c < 0:
        return

    if abs(b + np.sign(b) * np.sqrt(b ** 2 - 4 * a * c)) < epsilon:
        return

    delta_x = -2 * c / (b + np.sign(b) * np.sqrt(b ** 2 - 4 * a * c))

    x3 = x2 + delta_x
    k += 1
    x0 = x1
    x1 = x2
    x2 = x3
    while (abs(delta_x) >= epsilon and k <= max_iterations and delta_x <= 10 **8 ):
        h0 = x1 - x0
        h1 = x2 - x1
        f0 = horner(coefficients, x0)
        f1 = horner(coefficients, x1)
        f2 = horner(coefficients, x2)

        delta0 = (f1 - f0) / h0
        delta1 = (f2 - f1) / h1

        a = (delta1 - delta0) / (h1 + h0)
        b = a * h1 + delta1
        c = f2

        if b ** 2 - 4 * a * c < 0:
            return

        if abs(b - np.sqrt(b ** 2 - 4 * a * c)) < epsilon:
            return

        delta_x = -2 * c / (b + np.sign(b) * np.sqrt(b ** 2 - 4 * a * c))

        x3 = x2 + delta_x
        k += 1
        x0 = x1
        x1 = x2
        x2 = x3

    if(abs(delta_x) < epsilon):
        roots.append(x3)

    return roots


n = 4
coefficients = [1.0, -55.0/42.0, -42.0/42.0, 49.0/42.0, -6.0/42.0]
R = calculate_R(coefficients)
print(R)
x0, x1, x2 = 0.0, 2.0, 1.0
epsilon = 1e-10
max_iterations = 1000
roots_result = []
nr_incercari = 10000
i=0
while(len(roots_result) <n and i < nr_incercari ):
    x0 = round(random.uniform(-R, R), 2)
    x1 = round(random.uniform(-R, R), 2)
    x2 = round(random.uniform(-R, R), 2)
    root = muller_method(coefficients, x0, x1, x2, epsilon, max_iterations)
    if root and [round(root[0], 2)] not in roots_result:
        roots_result.append([round(root[0], 2)])
    i +=1
with open("roots.txt", "a") as file:
    for root in roots_result:
        file.write(str(root) + "\n")
print(roots_result)