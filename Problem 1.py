"""
Least-squares (20 pts) Code.Create  a  set  ofN=  100  points,xk,  uniformly  distributed in the [−1,+1] range.
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
    noise = np.random.normal(0, .1, None)
    noise_iteration = N_iteration + noise
    N.append(N_iteration)
    N_noise.append(noise_iteration)
    N_iteration += 0.02


# f(x) = sin(3x)
def f(x):
    return math.sin(3 * x)


count = 0
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

while count < 100:
    print(N[count], "|", f(N[count]), "|", N_noise[count], "|", f(N_noise[count]))
    ax1.scatter(N[count], f(N[count]), color="blue", s=10, label="f(xk)" if count == 0 else "")
    ax1.scatter(N_noise[count], f(N_noise[count]), color="green", s=10, label="fc(xk)" if count == 0 else "")
    count += 1

plt.legend()
plt.show()
