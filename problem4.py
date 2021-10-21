# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:20:30 2021

@author: Nathan Maupin

4. System of nonlinear equations (20 pts) Code.
Solve  the  following  nonlinear  sys-tem using 
iterative Newton,
"""

import numpy as np
import matplotlib.pyplot as plt
import math

x0, y0 = 3, 4
h = 1e-8
eps = 10e-6
A = []
b = []
error = 10
xL = []
yL = []
f1L = []
f2L = []
count = 0
count_array = []

def f1(x, y):
    return x**2 - y**2  - 1


def f2(x, y):
    return -x**2 + 2*y**3 - math.exp(-x*y)

def fprime(f, x, y):
    dx  = (f(x+h, y) - f(x-h, y))/(2*h)
    dy = (f(x, y+h) - f(x, y-h))/(2*h)
    return dx, dy

fig1 = plt.figure()
graph4 = fig1.add_subplot(111)
while error > eps:
    count_array.append(count)
    xL.append(x0)
    yL.append(y0)
    f1L.append(f1(x0, y0))
    f2L.append(f2(x0, y0))
    A.append(fprime(f1, x0, y0))
    A.append(fprime(f2, x0, y0))
    b.append(f1(x0, y0))    
    b.append(f2(x0, y0))
    x_diff, y_diff = np.linalg.solve(A, b) 
    v_diff = np.array([x_diff, y_diff]) # difference between vk1 and vk
    x0, y0 = x0 - x_diff, y0 - y_diff # calculating vk1 for x, y, z
    error = np.linalg.norm(v_diff, ord=2)
    A = []
    b = []
    count += 1
print(count)
count1 = 0
print("%18s" % "Iterations", "%18s" % "xk", "%18s" % "yk", "%18s" % "f1", "%18s" % "f2")
while count1 < count:
    print("%18s" % count_array[count1], "|", "%18s" % xL[count1], "|", "%18s" % yL[count1], "|", "%18s" % f1L[count1], "|", "%18s" % f2L[count1])
    count1 += 1

graph4.plot(count_array, xL, color="blue", label="xk")
graph4.plot(count_array, yL, color="red", label="yk")
graph4.plot(count_array, f1L, color="green", label="f1")
graph4.plot(count_array, f2L, color="orange", label="f2")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.show()
print("finished")
