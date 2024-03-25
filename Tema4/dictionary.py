import numpy as np

epsilon = 10 ** (-6)


def read_b_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())  # Extract dimension from the first line
        b = [float(line.strip()) for line in lines[1:]]  # Extract elements of vector b
    return n, b


def read_a_file_dict(file_path):
    matrix = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())
        for line in lines[1:]:
            values = line.strip().split(',')
            value = float(values[0])
            i = int(values[1])
            j = int(values[2]) if values[2] != '' else None
            if value != 0:
                if i not in matrix:
                    matrix[i] = {}
                if j is not None:
                    if j in matrix[i]:
                        matrix[i][j] += value
                    else:
                        matrix[i][j] = value
    return n, matrix


def sumGauss(matrix, i, x_c):
    sum = 0
    for key in matrix[i]:
        if key != i:
            sum += matrix[i][key] * x_c[key]
    return sum


def update_x(x_c, norm, rare_matrix, b, n):
    for i in range(n):
        element = x_c[i]
        x_c[i] = (b[i] - sumGauss(rare_matrix, i, x_c)) / rare_matrix[i][i]
        norm += (abs(x_c[i] - element)) ** 2

    return x_c, norm


def GaussSeidel(b, rare_matrix, n):
    x_c = np.zeros(n)
    k = 0
    k_max = 10000

    norm = 0
    x_c, norm = update_x(x_c, norm, rare_matrix, b, n)
    norm = np.sqrt(norm)
    k = k + 1

    while (norm >= epsilon) and (k < k_max) and (norm <= 10 ** 8):
        norm = 0
        x_c, norm = update_x(x_c, norm, rare_matrix, b, n)
        norm = np.sqrt(norm)
        k = k + 1

    if norm < epsilon:
        return x_c, k
    else:
        return None, k


def calculate_residual_norm(A_matrix, sol, b, n):
    sol = np.array(sol)

    Ax = np.zeros(n)
    for i, row in enumerate(A_matrix.keys()):
        for col, val in A_matrix[row].items():
            Ax[row] += val * sol[col]

    residual = np.abs(Ax - np.array(b))
    residualNorm = np.max(residual)

    return residualNorm


number = 1
file_path_for_b = f"b_{number}.txt"
file_path_for_a = f"a_{number}.txt"

n, b = read_b_file(file_path_for_b)
print("Dimension of b, n =", n)

n, matrix_A = read_a_file_dict(file_path_for_a)
print("Dimension of matrix A, n =", n)

for row, elements in matrix_A.items():
    if abs(matrix_A[row][row]) < epsilon:
        raise Exception("Det is 0 => cannot apply Gauss-Seidel")

solution, iterations = GaussSeidel(b, matrix_A, n)
if solution is None:
    print("Did not converge to a solution.")
    print("Number of iterations: ", iterations)
else:
    print("Solution: ", solution)
    print("Number of iterations: ", iterations)

if solution is not None:
    residual_norm = calculate_residual_norm(matrix_A, solution, b, n)
    if residual_norm < epsilon:
        print("Solution is accurate as residual norm is less than epsilon.")
    else:
        print("Solution may not be accurate as residual norm is greater than epsilon.")
