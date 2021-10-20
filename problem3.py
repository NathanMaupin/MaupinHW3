# -*- coding: utf-8 -*-
"""
@author: Nathan Maupin

3. System of nonlinear equations (20 pts) Code.Solve  the  following  nonlinear
  system,
"""
import math
import numpy as np

x0 = 0.1
y0 = 0.1
z0 = -0.1
h = 1e-8
vk = (x0, y0, z0)
A = []
b = []

def f1(x, y, z):
    return(3*z - math.sin(x*y) - 1/8)


def f2(x, y, z):
    return(x**2 - 19*(y+0.1)**3 + math.cos(z))


def f3(x, y, z):
    return(math.exp(y) + 20*x*z**2 + math.pi/4)


def fprime(f, x, y, z):
    dx  = (f(x+h, y, z) - f(x-h, y, z))/(2*h)
    dy = (f(x, y+h, z) - f(x, y-h, z))/(2*h)
    dz = (f(x, y, z+h) - f(z, y, z-h))/(2*h)
    return dx, dy, dz

#fprime(f1, 1, 1, 1)


while np.linalg.norm(vk) < h:
    A.extend(fprime(f1, x0, y0, z0), fprime(f2, x0, y0, z0), fprime(f3, x0, y0, z0))
    b.extend(f1(x0, y0, z0), f2(x0, y0, z0), f3(x0, y0, z0))
    print(A)
    print(b)
    print(test
    xbar = np.linalg.solve(A, b)
    print(xbar)
