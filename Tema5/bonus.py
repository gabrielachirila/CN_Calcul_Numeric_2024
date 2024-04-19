import copy
import numpy as np
epsilor = 10 ** (-5)

n = 3
A = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])


A_init = copy.deepcopy(A)
k_max = 10 ** 3

def find_index(i, j, n):
    line = i
    column = j
    if j > i:
        line = j
        column = i
    diff = 0
    for x in range(1, column+1):
        diff += x
    return line + (column*n) - diff

def createIn(n):
    In = np.zeros((n, n))
    for i in range(n):
        In[i][i] = 1.0
    return In

def calculateIndex(A, n):
    max_val = -1
    for i in range(1, n):
        for j in range(i):
            if abs(A[find_index(i, j, n)]) > max_val:
                max_val = abs(A[find_index(i, j, n)])
                p, q = i, j
    return p, q


def calculateUnghi(A, p, q):
    alpha = (A[find_index(p, p, n)] - A[find_index(q, q, n)]) / (2 * A[find_index(p, q, n)])
    if alpha >= 0:
        t = -alpha + np.sqrt(alpha ** 2 + 1)
    else:
        t = -alpha - np.sqrt(alpha ** 2 + 1)

    c = 1 / (np.sqrt(1 + t ** 2))
    s = t / (np.sqrt(1 + t ** 2))

    return t, c, s


def checkDiagonal(A, n): #o matrice e diagonala daca are doar elemente e diag principala
    for i in range(n):
        for j in range(n):
            if i != j and abs(A[find_index(i, j, n)]) > epsilor:
                return False
    return True


def calculateNewA(A, p, q, c, s, t):
    A_copy = np.copy(A)
    for j in range(n):
        if j != p and j != q:
            A_copy[find_index(p, j, n)] = c * A[find_index(p, j, n)] + s * A[find_index(q, j, n)]
            A_copy[find_index(q, j, n)] = -s * A[find_index(j, p, n)] + c * A[find_index(q, j, n)]

    for j in range(n):
        if j != p and j != q:
            A_copy[find_index(j, p, n)] = A_copy[find_index(p, j, n)]
            A_copy[find_index(j, q, n)] = A_copy[find_index(q, j, n)]

    A_copy[find_index(p, p, n)] = A[find_index(p, p, n)] + t * A[find_index(p, q, n)]
    A_copy[find_index(q, q, n)] = A[find_index(q, q, n)] - t * A[find_index(p, q, n)]

    A_copy[find_index(p, q, n)] = 0.0
    A_copy[find_index(q, p, n)] = 0.0
    return A_copy


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
    while(checkDiagonal(A, n ) == False and k < k_max):
        t, c, s = calculateUnghi(A, p, q)
        A = calculateNewA(A, p, q, c, s, t)
        U = calculate_U(U, p, q, s, c)
        p, q = calculateIndex(A, n)
        k = k + 1

    return A, U


A_result, U_result = JacobiAlg(A, n)
print(A_result)
print(checkDiagonal(A_result, n))
print(U_result)
# A_final = np.dot(U_result.T, A_init)
# A_final = np.dot(A_final, U_result)
# print(A_final)