import copy
import numpy as np

n = 4
epsilon = 10 ** (-6)

# A = np.array([[4.0, 2.0, 1.0],
#               [2.0, 5.0, 3.0],
#               [1.0, 3.0, 6.0]])

A = np.array([[1.0, 2.0, 3.0, 4.0],
              [2.0, 3.0, 4.0, 5.0],
              [3.0, 4.0, 5.0, 6.0],
              [4.0, 5.0, 6.0, 7.0]])

A_init = copy.deepcopy(A)
k_max = 10 ** 3


def createIn(n):
    In = np.zeros((n, n))
    for i in range(n):
        In[i][i] = 1
    return In


def calculateIndex(A, n):
    max_val = -1
    for i in range(1, n):
        for j in range(i):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    return p, q


def calculateAngle(A, p, q):
    alpha = (A[p][p] - A[q][q]) / (2 * A[p][q])
    if alpha >= 0:
        t = -alpha + np.sqrt(alpha ** 2 + 1)
    else:
        t = -alpha - np.sqrt(alpha ** 2 + 1)

    c = 1 / (np.sqrt(1 + t ** 2))
    s = t / (np.sqrt(1 + t ** 2))

    return t, c, s


# o matrice e diagonala daca are doar elemente e diag principala si restul sunt 0
def checkDiagonal(A, n):
    for i in range(n):
        for j in range(i):
            if abs(A[i][j]) > epsilon:
                return False
    return True


def calculateNewA(A, p, q, c, s, t):
    for j in range(n):
        if j != p and j != q:
            A[p][j] = c * A[p][j] + s * A[q][j]
            A[q][j] = -s * A[j][p] + c * A[q][j]

    for j in range(n):
        if j != p and j != q:
            A[j][p] = A[p][j]
            A[j][q] = A[q][j]

    A[p][p] = A[p][p] + t * A[p][q]
    A[q][q] = A[q][q] - t * A[p][q]

    A[p][q] = 0.0
    A[q][p] = 0.0
    return A


def calculate_U(U, p, q, s, c):
    U_veche = np.copy(U)
    for i in range(n):
        U[i][p] = c * U[i][p] + s * U[i][q]
        U[i][q] = -s * U_veche[i][p] + c * U[i][q]
    return U


def JacobiAlg(A, n):
    k = 0
    U = createIn(n)
    p, q = calculateIndex(A, n)
    while checkDiagonal(A, n) is False and k < k_max:
        t, c, s = calculateAngle(A, p, q)
        if abs(A[p][q]) > epsilon:
            A = calculateNewA(A, p, q, c, s, t)
        else:
            A = calculateNewA(A, p, q, c, s, t)
        U = calculate_U(U, p, q, s, c)
        p, q = calculateIndex(A, n)
        k = k + 1

    return A, U


A_result, U_result = JacobiAlg(A, n)
print("A_result:\n ", A_result)
print("U_result:\n ", U_result)
A_final = np.dot(U_result.T, A_init)
A_final = np.dot(A_final, U_result)
print("A_final:\n ", A_final)

result1 = np.dot(A_init, U_result)
result2 = np.dot(U_result, A_result)

for i in range(n):
    for j in range(n):
        err = abs(result1[i][j] - result2[i][j])
        if err > epsilon:
            print("Error: ", err)

result = np.allclose(np.dot(A_init, U_result), np.dot(U_result, A_result), atol=epsilon)

def calculate_norm(A_init, U, A_result):
    product_1 = np.dot(A_init, U)
    product_2 = np.dot(U, A_result)
    diff = product_1 - product_2
    norm = np.linalg.norm(diff)

    return norm


def calculate_norm2(matrix1, matrix2):
    diff = matrix1 - matrix2
    norm = np.linalg.norm(diff)

    return norm


print("norm: ", calculate_norm(A_init, U_result, A_result))


def cholesky_decomposition(matrix):
    L = np.linalg.cholesky(matrix)
    return L


def cholesky_factorization(matrix):
    A_current = np.copy(matrix)
    k = 0
    while k < k_max:
        L = cholesky_decomposition(A_current)
        A_next = np.dot(L.T, L)
        if calculate_norm2(A_next, A_current) < epsilon:
            break
        A_current = A_next
        k += 1

    return A_next


A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])


print("Factorizarea Cholesky a matricei A:")
print(cholesky_factorization(A))

A_init = np.array([[4, 1, -2],
                   [1, 10, 3],
                   [-2, 3, 12],
                   [1, 1, 4]])

# Dimensiunile matricii A_init
m, n = A_init.shape

U, S, VT = np.linalg.svd(A_init)

print("Valorile singulare ale matricei A:")
print(S)

rank_A = np.linalg.matrix_rank(A_init)
print("Rangul matricei A folosind biblioteca:", rank_A)

epsilon = 1e-10  # Definim o valoare mică pentru zero numeric
rank_A_eps = np.sum(S > epsilon)
print('Rangul matricii A:', rank_A_eps)

cond_A = np.linalg.cond(A_init)
print("Numărul de condiționare al matricei A folosind biblioteca:", cond_A)

cond_A1 = np.max(S) / np.min(S[S > epsilon])
print('Numărul de condiționare:', cond_A1)

S_i_values = [0 if abs(e) < epsilon else 1 / e for e in S]
S_i = np.diag(S_i_values)
A_i = VT.T[:, :rank_A] @ np.linalg.inv(np.diag(S[:rank_A])) @ U.T[:rank_A, :]
print('Pseudoinversa Moore-Penrose calculată:\n', A_i)

A_t_A = A_init.T @ A_init
A_t_A_inverse = np.linalg.pinv(A_t_A)
A_j = A_t_A_inverse @ A_init.T
print("A_j:", A_j)

matrix_norm = np.linalg.norm(A_i - A_j, 1)
print("Norm:", matrix_norm)

if matrix_norm > epsilon:
    print("Norma este mai mare decât epsilon.")
else:
    print("Norma este mai mică sau egală cu epsilon.")