import numpy as np

epsilon = 0.000005
n = 3
A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]])
b = [2, 2, 2]


def check_zero(number):
    if abs(number) < epsilon:
        raise Exception("Determinant is zero")


def LU_find(A):
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for p in range(n):
        for i in range(p, n):
            L[i, p] = A[i, p] - np.dot(L[i, :p], U[:p, p])

        for i in range(p + 1, n):
            U[p, i] = (A[p, i] - np.dot(L[p, :p], U[:p, i])) / L[p, p]

    np.fill_diagonal(U, 1)

    return L, U


def LU_find2(A):
    LU = np.zeros((n, n))

    for p in range(n):
        for i in range(p, n):
            diff = 0
            for k in range(p):
                if k == p:
                    diff = LU[i, k] * 1
                else:
                    diff += LU[i, k] * LU[k, p]
            LU[i, p] = A[i, p] - diff
            if i == p:
                check_zero(LU[i, p])
        for i in range(p + 1, n):
            diff = 0
            for k in range(p):
                if k == i:
                    diff = LU[p, k] * 1
                else:
                    diff += LU[p, k] * LU[k, i]
            LU[p, i] = (A[p, i] - diff) / LU[p, p]

    return LU


print("matricea LU \n", LU_find2(A))
LU = LU_find2(A)


def detCalculate(matrixLU):
    n = len(matrixLU)
    det = 1
    for i in range(n):
        det *= matrixLU[i][i]
    return det


print("\nDeterminantul matricei A:", detCalculate(LU))


def forwardSubst(LU, b):
    y = np.zeros(n)
    for i in range(n):
        dot_product = 0
        for j in range(i):
            dot_product += LU[i, j] * y[j]
        y[i] = (b[i] - dot_product) / LU[i, i]
    return y


def backwardSubs(LU, y):
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        dot_product = 0
        for j in range(i + 1, n):
            if i == j:
                dot_product += 1 * x[j]
            else:
                dot_product += LU[i, j] * x[j]
        x[i] = (y[i] - dot_product)
    return x


y = forwardSubst(LU,b)
x = backwardSubs(LU,y)
print("\nSolutia este:", x)


def calculate_euclidean_norm(x):
    sum_of_squares = 0
    for element in x:
        sum_of_squares += element ** 2
    norm = np.sqrt(sum_of_squares)
    return norm


def norm1(A, x_LU, b):
    a = np.zeros(len(b))
    for i in range(len(A)):
        for j in range(len(x_LU)):
            a[i] += A[i][j] * x_LU[j]
        a[i] -= b[i]

    norm = calculate_euclidean_norm(a)
    return norm


euclidean_norm = norm1(A, x, b)
print("\nNorma Euclidiana (1):", euclidean_norm)

if euclidean_norm < 10**(-9):
    print("Norma Euclidiana (1) este mai mica decat 10**(-9).")
else:
    print("Norma Euclidiana (1) este mai mare decat 10**(-9).")

x_lib = np.linalg.solve(A, b)
print("\nSoluția sistemului Ax = b calculata folosind biblioteca este:")
print(x_lib)

# inversa
A_inv = np.linalg.inv(A)
print("\nInversa matricei A calculate folosind biblioteca este:")
print(A_inv)


def norm2(x_LU, x_lib):
    a = x_LU.copy()
    for i in range(len(a)):
        a[i] -= x_lib[i]

    norm = calculate_euclidean_norm(a)
    return norm


print("\nNorma Euclidiana (2):", norm2(x, x_lib))

if norm2(x, x_lib) < 10**(-9):
    print("Norma Euclidiana (2) este mai mica decat 10**(-9).")
else:
    print("Norma Euclidiana (2) este mai mare decat 10**(-9).")


def norm3(x_LU, A_inv, b):
    a = np.zeros(len(b))
    for i in range(len(A_inv)):
        for j in range(len(b)):
            a[i] += A_inv[i][j] * b[j]
        y = x_LU[i] - a[i]
        a[i] = y

    norm = calculate_euclidean_norm(a)
    return norm


print("\nNorma Euclidiana (3):", norm3(x, A_inv, b))

if norm3(x, A_inv, b) < 10**(-9):
    print("Norma Euclidiana (3) este mai mica decat 10**(-9).")
else:
    print("Norma Euclidiana (3) este mai mare decat 10**(-9).")


#       ------------------------------------BONUS--------------------------------------

def LU_find_bonus(matrixA):
    n = len(matrixA)
    matrix = np.copy(matrixA)

    L_elements = np.zeros(n * (n + 1) // 2)
    U_elements = np.zeros(n * (n + 1) // 2)

    index_L = 0
    index_U = 0

    for p in range(n):
        for i in range(p, n):
            if p == 0:
                L_elements[index_L] = matrix[i, p]
                index_L += 1
            else:
                dif = 0
                nr_elemente_lipsa = 0
                for k in range(p):
                    nr_elemente_lipsa += k
                    dif += L_elements[p + (k * n) - nr_elemente_lipsa] * U_elements[p + (k * n) - nr_elemente_lipsa]
                L_elements[index_L] = matrix[i, p] - dif
                index_L += 1

        U_elements[index_U] = 1
        index_U += 1
        for i in range(p + 1, n):
            dif = 0
            nr_elemente_lipsa = 0
            for k in range(p):
                nr_elemente_lipsa += k
                dif += L_elements[i + (k * n) - nr_elemente_lipsa] * U_elements[p + (k * n) - nr_elemente_lipsa]
            U_elements[index_U] = (matrix[p, i] - dif) / L_elements[p + (p * n) - p]
            index_U += 1

    return L_elements, U_elements


def reconstruct_LU(L_elements, U_elements, n):
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    index_L = 0
    index_U = 0

    for i in range(n):
        for j in range(n):
            if i <= j:
                L[j, i] = L_elements[index_L]
                index_L += 1

    for i in range(n):
        for j in range(i, n):
            U[i, j] = U_elements[index_U]
            index_U += 1

    return L, U


A = np.array([[2.5, 2, 2], [5, 6, 5], [5, 6, 6.5]])

L_elements, U_elements = LU_find_bonus(A)
print("\nL", L_elements)
print("U", U_elements)
L, U = reconstruct_LU(L_elements,U_elements, len(A))
print("Matricea L\n", L)
print("Marticea U\n", U)
b = np.array([2, 2, 2])


def calculate_y(L_vector, b):
    y = np.zeros(n)
    for i in range(n):
        nr_elemente_lipsa = 0
        diff = 0
        for j in range(i):
            nr_elemente_lipsa += j
            diff += L_vector[i + (j*n) - nr_elemente_lipsa] * y[j]
        y[i] = (b[i] - diff) / L_vector[n * i - nr_elemente_lipsa]
    return y


y2 = calculate_y(L_elements, b)
print("\ny2 = ", y2)
print("y = ", y)


def calculate_x(U_vector, y):
    x = np.zeros(n)
    nr_elemente_lipsa = 0
    for i in range(n):
        nr_elemente_lipsa += i
    for i in range(n - 1, -1, -1):
        nr_elemente_lipsa -= i
        sum_upper = 0
        for j in range(i + 1, n):
            sum_upper += U_vector[i * n + j - (nr_elemente_lipsa+i)] * x[j]
        x[i] = (y[i] - sum_upper) / U_vector[i * n - nr_elemente_lipsa]
    return x


x2 = calculate_x(U_elements, y2)
print("\nx2 = ", x2)
print("x = ", x)
