import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

n = 1000
x = np.random.normal(3, 5, size=n)
y =  -15 * x + 20 + np.random.normal(0, 3, size=n)

X = np.sum(x)
Y = np.sum(y)
w1 = (n * np.dot(x, y) - Y * X) / (n * np.dot(x, x) - X ** 2)
w0 = (Y - w1 * X) / n
w1, w0 = round(w1, 2), round(w0, 2)
print(f"y  = {w1} x + {w0}")