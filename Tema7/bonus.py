def horner(coefficients, v):
    result = 0
    for coefficient in coefficients:
        result = result * v + coefficient
    return result

def newton_fourth_order(coefficients_f, coefficients_deriv, x0, epsilon=1e-6, max_iterations=100):
    x = x0
    for i in range(max_iterations):
        fx = horner(coefficients_f, x)
        fdx = horner(coefficients_deriv, x)
        y = x - (fx / fdx)
        miu_n = horner(coefficients, y) / horner(coefficients, x)
        xn1 = x - ((1 + miu_n ** 2) / (1 - miu_n) )* (fx / fdx)
        error = abs(xn1 - x)
        if error < epsilon:
            return xn1
        x = xn1
    raise ValueError("Nu s-a convergat în numărul maxim de iterații")

def compute_derivative(coefficients):
    # Calculăm derivatele polinomului dat
    n = len(coefficients) - 1
    derivative_coefficients = [coefficients[i] * (n - i) for i in range(n)]
    return derivative_coefficients

# Exemplu de utilizare:
coefficients = [1.0, -55.0/42.0, -42.0/42.0, 49.0/42.0, -6.0/42.0]
coefficients_deriv = compute_derivative(coefficients)
x0 = 0.0
root = newton_fourth_order(coefficients,coefficients_deriv, x0)
print("Rădăcina găsită:", root)