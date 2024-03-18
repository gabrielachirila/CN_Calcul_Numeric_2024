import math
import numpy as np

epsilon = 0.0001
n = 3
A_init = np.array([[0, 0, 4], [1, 2, 3], [0, 1, 2]])
# A_init = np.random.rand(n, n)
s = [3, 2, 1]

# n = 4
# A_init = np.array([[1, 2, 1, 1], [1, 4, -1, 7], [4, 9, 5, 11], [1, 0, 6, 4]])
# s = [0, 20, 18, 1]


# ex1
def create_b(n, A):
    b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            b[i] += A[i][j] * s[j]
    return b


b_init = create_b(n, A_init)
print("b_init = ", b_init)


# ex2
def create_I_n(n):
    I_n = np.zeros((n, n))
    for i in range(n):
        I_n[i][i] = 1

    return I_n


def Householder(n, A):
    Q_init = create_I_n(n)
    b = create_b(n, A)
    A_copy = np.copy(A)
    for i in range(n):
        P_i = generate_P_i(i, A_copy)
        A_copy = np.dot(P_i, A_copy)
        Q_init = np.dot(P_i, Q_init)
        b = np.dot(P_i, b)

    return Q_init, A_copy, b


def create_u(n, k, r, A):
    u = np.zeros(n)
    u[r] = A[r][r] - k
    for i in range(r + 1, n):
        u[i] = A[i][r]
    return u


def generate_P_i(r, A):
    sigma = 0
    for j in range(r, n):
        sigma += A[j][r] ** 2

    if A[r][r] < 0:
        k = math.sqrt(sigma)
    else:
        k = -math.sqrt(sigma)

    beta = sigma - (k * A[r][r])
    u = create_u(n, k, r, A)

    v = np.zeros((n, n))
    for j in range(r, n):
        for k in range(r, n):
            v[j][k] = u[j] * u[k]

    I_n = create_I_n(n)
    P = I_n - ((1 / beta) * v)

    return P


Q, R, b = Householder(n, A_init)
print("\nQ_T = ", Q)
print("\nR = ", R)
print("\nQ_T * b = ", b)


# ex3
R_inv = np.linalg.inv(R)
x_Householder = np.dot(R_inv, b)
print("\nSoluția sistemului Ax = b este x = ", x_Householder)


def calculate_euclidean_norm(x):
    sum_of_squares = 0
    for element in x:
        sum_of_squares += element ** 2
    norm = np.sqrt(sum_of_squares)
    return norm


def norm(x, x_lib):
    a = x.copy()
    for i in range(len(a)):
        a[i] -= x_lib[i]

    norm = calculate_euclidean_norm(a)
    return norm


Q_lib, R_lib = np.linalg.qr(A_init)
b_lib = np.dot(A_init, s)
x_QR = np.linalg.solve(R_lib, np.dot(Q_lib.T, b_lib))
print("Soluția sistemului Ax = b calculata folosind librarie este:", x_QR)

print("\n")


def verify_norm(norm, i):
    if norm < 10 ** (-9):
        print("Norma (", i, ") este mai mica decat 10**(-6).")
    else:
        print("Norma (", i, ") este mai mare decat 10**(-6).")


norma1 = norm(x_Householder, x_QR)
verify_norm(norma1, 1)

# ex4
p = np.dot(A_init, x_Householder)
norma2 = norm(p, b_init)
verify_norm(norma2, 2)

p = np.dot(A_init, x_QR)
norma3 = norm(p, b_init)
verify_norm(norma3, 3)

x_1 = norm(x_Householder, s)
x_2 = calculate_euclidean_norm(s)
norma4 = x_1 / x_2
verify_norm(norma4, 4)

x_1 = norm(x_QR, s)
norma5 = x_1 / x_2
verify_norm(norma5, 5)


# ex5
def check_zero(number):
    if abs(number) < epsilon:
        raise Exception("Determinant is zero")


def backwardSubs(R, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    return x


def find_A_inv(Q, R):
    for i in range(n):
        check_zero(R[i][i])

    A_inv = np.zeros((n, n))
    b1 = np.zeros(n)

    for j in range(n):
        for k in range(n):
            b1[k] = -Q[k][j]
        x = backwardSubs(R, b1)
        A_inv[:, j] = -x

    return A_inv


A_inv_lib = np.linalg.inv(A_init)
print("\nInversa matricei A calculate folosind biblioteca este:")
print(A_inv_lib)

A_inv = find_A_inv(Q, R)
print("\nInversa matricei A calculate manual este:")
print(A_inv)

print("\n")
norma6 = np.linalg.norm(np.abs(A_inv_lib - A_inv))
verify_norm(norma6, 6)

print("\n************************************ BONUS ************************************")

Q = -Q.T
R = -R


def bonus(A):
    k = 1
    A_array = []
    Q, R, b = Householder(n, A)
    Q = Q.T
    A_array.append(A)
    A_array.append(dot_RQ(R, Q))

    while np.linalg.norm(np.abs(A_array[k] - A_array[k - 1]), 2) > epsilon and k < 1000:
        k += 1
        Q, R, b = Householder(n, A_array[k - 1])
        Q = Q.T
        A_array.append(dot_RQ(Q, R))
        k += 1
        A_array.append(dot_RQ(R, Q))

    return k


def dot_RQ(R, Q):
    result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(i, n):
                result[i][j] += R[i][k] * Q[k][j]
                result[j][i] += R[i][k] * Q[k][j]

    return result


A_sim = [[0.27839766, 0.43264983, 0.52901993], [0.91962197, 0.32588897, 0.72812641], [0.44538844, 0.82635532, 0.36898862]]
print("k FINAL= ", bonus(A_sim))
