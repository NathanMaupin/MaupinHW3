# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:19:44 2021

@author: Nathan Maupin
"""

"""
1. Least-squares (20 pts) Code.Create  a  set  ofN=  100  points,xk,  uniformly  distributed in the [âˆ’1,+1] range.
Compute the function f(x) = sin(3x) at the xk data points.  Corrupt the f(xk) values by adding a Gaussian noise with
standard deviation Ïƒ= 0.08.  Plo f(xk) and fc(xk).  Perform the least-squares of fc(xk) using the fitting functions,
"""
import math
import numpy as np
import matplotlib.pyplot as plt


N_iteration = -1
N = []
N_noise = []
deviation = 0.08

# Creates array of 100 points for N
while N_iteration <= 1:
    N_iteration = round(N_iteration, 2)
    N.append(N_iteration)
    N_iteration += 0.02


# f(x) = sin(3x)
def f(x):
    return math.sin(3 * x)


# Plots f(xk) and fc(xk)
count = 0
fig1 = plt.figure()
graph1 = fig1.add_subplot(111)
noise = np.random.normal(0, .08, None)
while count < 100:
    graph1.scatter(N[count], f(N[count]), color="blue", s=10, label="f(xk)" if count == 0 else "")
    graph1.scatter(N[count], f(N[count]) + noise, color="orange", s=10, label="fc(xk)" if count == 0 else "")
    count += 1
plt.title("f(xk) vs. fc(xk)")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.show()


#Least Square with f1(x)
fig2 = plt.figure()
graph2 = fig2.add_subplot(111)
a1 = []
b1 = []
for n1 in N:
    a1.append([1, n1, 2*n1**2 - 1, 4*n1**3 - 3*n1])
    b1.append(f(n1) + noise)

c1, residuals, rank, s = np.linalg.lstsq(a1, b1, rcond=None)


def f1(x):
    return c1[0] + c1[1]*x +c1[2]*(2*x**2 - 1) + c1[3]*(4*x**3 - 3*x)

#Least Square with f2(x)
a2 = []
b2 = []
for n2 in N:
    a2.append([math.sin(n2**2), 1 - math.cos(n2), math.cos(n2)*math.sin(n2), (2-n2)/(2+n2)])
    b2.append(f(n2) + noise)

c2, residuals, rank, s = np.linalg.lstsq(a2, b2, rcond=None)

def f2(x):
    return c2[0]*math.sin(x**2) + c2[1]*(1 - math.cos(x)) +c2[2]*(math.cos(x)*math.sin(x)) + c2[3]*((2-x)/(2+x))

f1y = []
f2y = []
for n in N:
    f1y.append(f1(n))
    f2y.append(f2(n))
    
plt.title("f1(x) vs. f2(x)")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
graph2.scatter(N, f1y, s=10, color="black", label="f1(x)")
graph2.scatter(N, f2y, s=10, color="red", label="f2(x)")
plt.legend()
plt.show()

#Calculating Norms

norm11 = np.linalg.norm(np.array(b1) - np.array(f1y), ord=1)
norm21 = np.linalg.norm(np.array(b2) - np.array(f2y), ord=1)
norm12 = np.linalg.norm(np.array(b1) - np.array(f1y), ord=2)
norm22 = np.linalg.norm(np.array(b2) - np.array(f2y), ord=2)
norm1inf = np.linalg.norm(np.array(b1) - np.array(f1y), ord=np.inf)
norm2inf = np.linalg.norm(np.array(b2) - np.array(f2y), ord=np.inf)
print("******Problem 1******")
print("f1(x) 1st Norm:", norm11)
print("f2(x) 1st Norm:", norm21)
print("f1(x) 2nd Norm:", norm12)
print("f2(x) 2nd Norm:", norm22)
print("f1(x) Infinite Norm:", norm1inf)
print("f2(x) Infinite Norm:", norm2inf)



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
eps =  10e-9
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


while error > eps:
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
print()
print("******Problem 3******")
print(count, "iterations")
plt.plot(count_array, error_array, color="red")
plt.show()



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
count1 = 0
print()
print("******Problem 4******")
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
