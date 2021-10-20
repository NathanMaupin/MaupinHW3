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
print("f1(x) 1st Norm:", norm11)
print("f2(x) 1st Norm:", norm21)
print("f1(x) 2nd Norm:", norm12)
print("f2(x) 2nd Norm:", norm22)
print("f1(x) Infinite Norm:", norm1inf)
print("f2(x) Infinite Norm:", norm2inf)
