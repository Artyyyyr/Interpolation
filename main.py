import matplotlib.pyplot as plt
import numpy as np


def fun(x):
    a = 5
    return np.sin(a/2 * x) + np.cbrt(x * a)


step = 0.01
start = 0
end = 8

x = np.arange(start, end, step)
y = fun(x)


def get_c(x, y):
    size = len(y)
    c = np.zeros([size, size])
    c[:, 0] = y

    for j in range(1, size):
        for i in range(size - j):
            c[i][j] = (c[i + 1][j - 1] - c[i][j - 1]) / (x[i + j] - x[i])

    return c


print("Newton method\n")


def newton_poly(c, x_data, x):
    size = len(x_data) - 1
    p = c[size]
    print("p1 = " + str(p))
    for k in range(1, size + 1):
        p = c[size - k] + (x - x_data[size - k]) * p
        if k == size:
            print("y = " + str(c[size - k]) + " + ( x - " + str(x_data[size - k]) + " ) * " + "p" + str(k))
        else:
            print("p" + str(k+1) + " = " + str(c[size - k]) + " + ( x - " + str(x_data[size - k]) + " ) * " + "p" + str(k))
    return p


X = np.array([1, 2, 3, 6, 8])
Y = fun(X)

a_s = get_c(X, Y)[0, :]
y_new = newton_poly(a_s, X, x)

plt.figure("Newton method")
plt.plot(X, Y, 'bo')
plt.plot(x, y_new)

plt.plot(x, y)

plt.figure("Newton method error")

error = np.abs(y - y_new)
plt.plot(X, np.zeros(len(X)), 'bo')
plt.plot(x, error)

# a + b * (x - x0) + c * (x - x0)^2 + d * (x - x0)^3
c_a = Y[:len(Y)-1]
b = np.zeros(len(X)-1)
c = np.zeros(len(X)-1)
d = np.zeros(len(X)-1)

c[0] = 0

# b1 c1 d1 b2 c2 d2 ... bn cn dn = S
mat = np.zeros((3*(len(X)-1), 3*(len(X)-1) + 1))

for i in range(len(X)-1):
    mat[i][3 * i] = X[i + 1] - X[i]
    mat[i][3 * i + 1] = pow(X[i + 1] - X[i], 2)
    mat[i][3 * i + 2] = pow(X[i + 1] - X[i], 3)
    mat[i][len(mat[i])-1] = Y[i + 1] - Y[i]

for i in range(len(X)-2):
    mat[i + len(X)-1][i * 3] = 1
    mat[i + len(X) - 1][i * 3 + 1] = 2 * (X[i + 1] - X[i])
    mat[i + len(X) - 1][i * 3 + 2] = 3 * pow(X[i + 1] - X[i], 2)
    mat[i + len(X) - 1][i * 3 + 3] = -1

for i in range(len(X)-2):
    mat[i + 2 * len(X) - 3][i * 3 + 1] = 2
    mat[i + 2 * len(X) - 3][i * 3 + 2] = 6 * (X[i + 1] - X[i])
    mat[i + 2 * len(X) - 3][i * 3 + 4] = -2

mat[len(mat) - 2][1] = 2

mat[len(mat) - 1][len(mat[len(mat) - 1]) - 2] = 6 * (X[len(X) - 1] - X[len(X) - 2])
mat[len(mat) - 1][len(mat[len(mat) - 1]) - 3] = 2

a = []
b = []
for i in range(len(mat)):
    a.append([*mat[i][:len(mat)]])
    b.append(mat[i][len(mat[i]) - 1])
a = np.array(a)
b = np.array(b)

c = np.linalg.solve(a, b)
plt.figure("Section cub equations method")
plt.plot(x, y)
plt.plot(X, Y, 'bo')

for i in range(len(X)-1):
    if i == 0:
        X_sec = np.arange(X[i] - X[0] + start, X[i + 1] + step, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        plt.plot(X_sec, Y_sec)
    elif i == len(X) - 2:
        X_sec = np.arange(X[i], X[i + 1] - X[len(X) - 1] + end, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        plt.plot(X_sec, Y_sec)
    else:
        X_sec = np.arange(X[i], X[i + 1] + step, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        plt.plot(X_sec, Y_sec)

plt.figure("Section cub ecuations method error")

for i in range(len(X)-1):
    if i == 0:
        X_sec = np.arange(X[i] - X[0] + start, X[i + 1] + step, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        error = np.abs(Y_sec - fun(X_sec))
        plt.plot(X_sec, error)
    elif i == len(X) - 2:
        X_sec = np.arange(X[i], X[i + 1] - X[len(X) - 1] + end, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        error = np.abs(Y_sec - fun(X_sec))
        plt.plot(X_sec, error)
    else:
        X_sec = np.arange(X[i], X[i + 1] + step, step)
        Y_sec = c_a[i] + c[3 * i] * (X_sec - X[i]) + c[3 * i + 1] * pow(X_sec - X[i], 2) + c[3 * i + 2] * pow(X_sec - X[i], 3)
        error = np.abs(Y_sec - fun(X_sec))
        plt.plot(X_sec, error)
plt.plot(X, np.zeros(len(X)), 'bo')

print("\nSection cub ecuations method\n")
for i in range(len(X)-1):
    print("Section " + str(i + 1))
    print("y = " + str(c_a[i]) + " + (" + str(c[i*3]) + ") * (x - " + str(X[i]) + ") + (" + str(c[i*3 + 1]) + ") * (x - " + str(X[i]) + ")^2 + (" + str(c[i*3 + 2]) + ") * (x - " + str(X[i]) + ")^3")

plt.show()
