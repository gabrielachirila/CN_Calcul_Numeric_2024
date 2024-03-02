import random
import numpy as np
import math


# Exercitiul 1: precizia masina

def machine_precision(initNumber):
    u = 10 ** (-initNumber)
    p = 0
    while 1 + u != 1:
        u /= 10
        p += 1
    return u * 10, p - 1


u, p = machine_precision(0)

print("-----Exercitiul 1-----")
print("u =", u)
print("p =", p)

# Exercitiul 2: asociativitate


def sum_ex2(u):
    x = 1.0
    y = 1.0
    z = u/10
    return (x+y)+z == x + (y+z)


print("-----Exercitiul 2-----")

if sum_ex2(u):
    print("Pentru numerele date operatia de adunare este neasociativa")
else:
    print("Pentru numerele date operatia de adunare este asociativa")


print("Exemplu pentru care operatia de inmultire este neasociativa")

def product_ex2():
    prods1 = 1
    prods2 = 1
    x = 0
    y = 0
    z = 0
    while prods1 == prods2:
        x = random.random()
        y = random.random()
        z = random.random()

        prods1 = (x * y) * z
        prods2 = x * (y * z)
    print("x=", x)
    print("y=", y)
    print("z=", z)
    print('(x*y)*z = {:.20}'.format(prods1))
    print('x*(y*z) = {:.20}'.format(prods2))


product_ex2()

# Exercitiul 3: Evaluarea functiei tangenta

print("-----Exercitiul 3-----")


def t(i, a):
    pi = 1
    qi = 1
    if i == 4:
        pi = (105 * a) - 10*(a**3)
        qi = 105 - (45*(a**2)) + (a**4)
    if i == 5:
        pi = (945*a) - (105*(a**3)) + (a**5)
        qi = 945 - (420*(a**2)) + (15*(a**4))
    if i == 6:
        pi = (10395*a) - (1260*(a**3)) +(21*(a**5))
        qi = 10395 - (4725*(a**2)) + (210* (a**4)) - (a**6)
    if i == 7:
        pi = (135135*a) - (17325*(a**3)) + (378*(a**5)) - (a**7)
        qi = 135135 - (62370*(a**2)) + (3150 * (a**4)) - (28*(a**6))
    if i == 8:
        pi = (2027025*a) - (270270*(a**3)) + (6930*(a**5)) - (36*(a**7))
        qi = 2027025 - (945945*(a**2)) + (51975*(a**4)) - (630*(a**6)) + (a**8)
    if i == 9:
        pi = (34459425*a) - (4729725*(a**3)) + (135135*(a**5)) - (990*(a**7)) + (a**9)
        qi = 34459425 - (16216200*(a**2)) + (945945*(a**4)) - (13860*(a**6)) + (45*(a**8))
    return pi/qi


def ex3():
    error = {'4': [], '5': [], '6': [], '7': [], '8': [], '9': []}
    random_numbers = np.random.uniform(-np.pi / 2, np.pi / 2, 10000)
    for number in random_numbers:
        v_ex = math.tan(number)
        val_t4 = t(4,number)
        val_t5 = t(5, number)
        val_t6 = t(6,number)
        val_t7 = t(7, number)
        val_t8 = t(8, number)
        val_t9 = t(9,number)
        error['4'].append(abs(val_t4 - v_ex))
        error['5'].append(abs(val_t5 - v_ex))
        error['6'].append(abs(val_t6 - v_ex))
        error['7'].append(abs(val_t7 - v_ex))
        error['8'].append(abs(val_t8 - v_ex))
        error['9'].append(abs(val_t9 - v_ex))

    errors = {}
    for key in error.keys():
        errors[key] = (np.mean(error[key]))

    sort_errors = sorted(errors.items(), key=lambda x: x[1])
    print("Erori sortate", sort_errors)
    print("Top 1:", sort_errors[0])
    print("Top 2:", sort_errors[1])
    print("Top 3:", sort_errors[2])


ex3()

# aproximarea functiilor sin si cos


def s(i, a):
    return t(i, a)/math.sqrt(1 + t(i, a)**2)


def c(i, a):
    return 1/math.sqrt(1 + t(i, a)**2)


print("Pentru valoarea i = 9 si a = pi/6")
print("tangenta:", t(9, np.pi/6))
sin = s(9, np.pi/6)
cos = c(9, np.pi/6)
print("sin:", sin)
print("cos:", cos)
print("sin/cos:", sin/cos)

