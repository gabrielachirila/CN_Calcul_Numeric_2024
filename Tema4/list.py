import numpy as np

epsilon = 10**(-6)


def read_b_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())  # Extract dimension from the first line
        b = [float(line.strip()) for line in lines[1:]]  # Extract elements of vector b
    return n, b


def read_a_file_list(file_path):
    rare_matrix = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n = int(lines[0].strip())

        for _ in range(n):
            rare_matrix.append([])

        for line in lines[1:]:
            values = line.strip().split(',')
            value = float(values[0])
            i = int(values[1])
            j = int(values[2]) if values[2] != '' else None
            if value != 0:
                if j is not None:
                    found = False
                    for index, (x, col) in enumerate(rare_matrix[i]):
                        if col == j:
                            rare_matrix[i][index] = (x + value, col)
                            found = True
                            break
                    if not found:
                        rare_matrix[i].append((value, j))
    return n, rare_matrix


def sum_Gauss(rare_matrix, i, x_c):
    sum = 0
    for val, col in rare_matrix[i]:
        if col != i:
            sum += val * x_c[col]
    return sum


def update_x(x_c, norm, rare_matrix, b, n):
    for i in range(n):
        element = x_c[i]
        for val, col in rare_matrix[i]:
            if col == i:
                x_c[i] = (b[i] - sum_Gauss(rare_matrix, i, x_c)) / val
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

    while (norm >= epsilon) and (k < k_max) and (norm <= 10**8):
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
    for i in range(n):
        for val, col in A_matrix[i]:
            Ax[i] += val * sol[col]

    residual = np.abs(Ax - np.array(b))
    residualNorm = np.max(residual)

    return residualNorm


number = 1
file_path_for_b = f"b_{number}.txt"
file_path_for_a = f"a_{number}.txt"
n, b = read_b_file(file_path_for_b)

print("Dimension of b, n =", n)

n, matrix_A = read_a_file_list(file_path_for_a)

print("Dimension of matrix A, n =", n)

for i in range(n):
    for val, col in matrix_A[i]:
        if col == i:
            if (abs(val) < epsilon) and (col == i):
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