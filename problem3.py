# -*- coding: utf-8 -*-
"""
@author: Nathan Maupin

3. System of nonlinear equations (20 pts) Code.Solve  the  following  nonlinear
  system,
"""
import math
import numpy as np
import matplotlib.pyplot as plt

x0 = 0.1
y0 = 0.1
z0 = -0.1
h = 1e-8
vk = np.array([x0, y0, z0])
A = []
b = []
error = 999999999
error_array = []
count = 0
count_array = []


def f1(x, y, z):
    return -1*(3*z - math.sin(x*y) - (1/8))


def f2(x, y, z):
    return -1*(x**2 - 19*(y+0.1)**3 + math.cos(z))


def f3(x, y, z):
    return -1*(math.exp(-y) + 20*x*(z**2) + (math.pi/4))


# x, y, z derivatives of each function
def fprime(f, x, y, z):
    dx  = (f(x+h, y, z) - f(x-h, y, z))/(2*h)
    dy = (f(x, y+h, z) - f(x, y-h, z))/(2*h)
    dz = (f(x, y, z+h) - f(x, y, z-h))/(2*h)
    return dx, dy, dz


while error > h:
    count_array.append(count)
    A.append(fprime(f1, x0, y0, z0))
    A.append(fprime(f2, x0, y0, z0))
    A.append(fprime(f3, x0, y0, z0))    
    b.append(f1(x0, y0, z0))    
    b.append(f2(x0, y0, z0))
    b.append(f3(x0, y0, z0))
    x_diff, y_diff, z_diff = np.linalg.solve(A, b) 
    v_diff = np.array([x_diff, y_diff, z_diff]) # difference between vk1 and vk
    x0, y0, z0 = x0 - x_diff, y0 - y_diff, z0 - z_diff # calculating vk1 for x, y, z
    error = np.linalg.norm(v_diff, ord=2)
    error_array.append(error)
    #print("error:", error)
    A = []
    b = []
    count += 1
    
print(count, "iterations")
plt.plot(count_array, error_array, color="red")
plt.show()
