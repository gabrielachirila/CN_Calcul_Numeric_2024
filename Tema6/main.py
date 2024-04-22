import numpy as np
import matplotlib.pyplot as plt


def function1(x):
    return x ** 4 - 12 * x ** 3 + 30 * x ** 2 + 12


def generate_interpolation_nodes(x0, xn, n):
    h = (xn - x0) / n

    nodes = [x0 + i * h for i in range(1, n)]

    return nodes


def interpolate_function_values(nodes, func):
    values = [func(x) for x in nodes]

    return values


def calculate_differences(x, y):
    n = len(x)
    A_schema = np.zeros((n, n))
    A_y = np.zeros(n)
    A_schema[:, 0] = y

    for j in range(1, n):
        for i in range(j, n):
            A_schema[i, j] = (A_schema[i, j - 1] - A_schema[i - 1, j - 1])

    for i in range(n):
        A_y[i] = A_schema[i, i]

    return A_schema, A_y


def interpolationNewton(x, y, target_x, actual_value):
    n = len(x)
    result = y[0]

    h = (x[n - 1] - x[0]) / (n - 1)
    t = (target_x - x[0]) / h
    s = 1
    schema, y_Aitken = calculate_differences(x, y)

    for i in range(1, n):
        if i == 1:
            s *= t
        else:
            s *= (t - i + 1) / i
        result += y_Aitken[i] * s

    error = abs(result - actual_value)

    return result, error


def Horner_schema(coefficients, target):
    d = coefficients[0]
    length = len(coefficients)

    for i in range(1, length):
        d = coefficients[i] + d * target

    return d


def least_squares_interpolation(x_values, y_values, target_x, polynomial_degree, actual_value):
    n = len(x_values)
    B = list()
    f = list()

    for i in range(polynomial_degree + 1):
        B_line = list()
        for j in range(polynomial_degree + 1):
            val = 0
            for k in range(n):
                val += x_values[k] ** (i + j)
            B_line.append(val)

        B.append(B_line)

    for i in range(polynomial_degree + 1):
        val = 0
        for k in range(n):
            val += y_values[k] * (x_values[k] ** i)
        f.append(val)

    coefficients = np.linalg.solve(np.array(B), np.array(f))
    coefficients = list(reversed(coefficients))

    result = Horner_schema(coefficients, target_x)

    sum_of_errors = 0
    for i in range(n):
        P_x = Horner_schema(coefficients, x_values[i])
        sum_of_errors += abs(abs(y_values[i]) - abs(P_x))

    error = abs(abs(result) - abs(actual_value))

    return result, error, sum_of_errors


def function_graphic(x_values, y_values, number):
    plt.plot(x_values, y_values, label='f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f"Function graphic {number}")
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_interpolation_errors(x_values, y_values, target_x, polynomial_degree, function_or_actual_value):
    if callable(function_or_actual_value):
        actual_value = function_or_actual_value(target_x)
    else:
        actual_value = function_or_actual_value

    Ln, error_newton = interpolationNewton(x_values, y_values, target_x, actual_value)
    LSI_result, error_LSI, sum_of_errors_LSI = least_squares_interpolation(x_values, y_values,
                                                                           target_x, polynomial_degree, actual_value)

    return Ln, LSI_result, sum_of_errors_LSI, error_newton, error_LSI


def interpolate_and_compare(x_values, y_values, target_x, grad, function_or_actual_value):
    Ln, LSI, LSI_sum_of_errors, error_newton, error_least_squares = (
        calculate_interpolation_errors(x_values, y_values, target_x, grad, function_or_actual_value))
    print("x =", x_values)
    print("y = f(x) =", y_values)
    print("target x =", target_x)
    print("L_n(x) =", Ln)
    print("P_m(x) =", LSI)
    if callable(function_or_actual_value):
        print("Actual function value at x =", target_x, "is:", function_or_actual_value(target_x))
    else:
        print("Actual function value at x =", target_x, "is:", function_or_actual_value)
    print("Error (Newton's method):", error_newton)
    print("Error (LSI):", error_least_squares)
    print("Sum of errors (LSI):", LSI_sum_of_errors, "\n")


m = 5

x1 = np.array([0, 1, 2, 3, 4, 5])
y1 = np.array([50, 47, -2, -121, -310, -545])
target1 = 1.5
actual_value1 = 30.3125
interpolate_and_compare(x1, y1, target1, m, actual_value1)
function_graphic(x1, y1, 1)

a = 1
b = 5
# x2 = np.linspace(a, b, 10)
x2 = generate_interpolation_nodes(a, b, 10)
y2 = interpolate_function_values(x2, function1)
target2 = 1.5
interpolate_and_compare(x2, y2, target2, m, function1)
function_graphic(x2, y2, 2)




