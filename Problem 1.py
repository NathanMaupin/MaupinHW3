"""
1. Least-squares (20 pts) Code.Create  a  set  ofN=  100  points,xk,  uniformly  distributed in the [−1,+1] range.
Compute the function f(x) = sin(3x) at the xk data points.  Corrupt the f(xk) values by adding a Gaussian noise with
standard deviation σ= 0.08.  Plo f(xk) and fc(xk).  Perform the least-squares of fc(xk) using the fitting functions,
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
    #noise = np.random.normal(0, .1, None)
    #noise_iteration = N_iteration + noise
    N.append(N_iteration)
    #N_noise.append(noise_iteration)
    N_iteration += 0.02


# f(x) = sin(3x)
def f(x):
    return math.sin(3 * x)


# Plots f(xk) and fc(xk)
count = 0
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
noise = np.random.normal(0, .1, None)
while count < 100:
    ax1.scatter(N[count], f(N[count]), color="blue", s=10, label="f(xk)" if count == 0 else "")
    ax1.scatter(N[count], f(N[count]) + noise, color="green", s=10, label="fc(xk)" if count == 0 else "")
    count += 1
plt.title("f(xk) vs. fc(xk)")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.legend()
plt.show()

a = []
b = []
for n in N:
    a.append([1, n, 2*n**2 - 1, 4*n**3, -3*n])
    b.append(f(n) + noise)

x = np.linalg.lstsq(a, b, rcond=None)
print("###########@###")
print(x)
    


