import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets

n = 1000
x = np.random.normal(3, 5, size = n)
y =  -15 * x + 20 + np.random.normal(0, 3, size=n)

def MSE_loss(w0, w1, x, y):
    pred = w0 + w1 * x
    loss = np.sum((pred - y) ** 2) / n
    return loss

#Generate a grid of w0 and w1 values round the real w0 = 20 and w1=-15
w0_vals = np.linspace(0, 40, 100)
w1_vals = np.linspace(-30, 0, 100)

#calculate MSE loss for each point of this grid
#Log-scale is used only for better visualization
Z = np.zeros((len(w0_vals), len(w1_vals)))
for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        Z[i, j] = np.log(MSE_loss(w0_vals[i], w1_vals[j], x, y))

#3d plot of MSE loss in log-scale
fig = plt.figure(figsize=(20, 15))
ax  = fig.add_subplot(111, projection='3d')
B0, B1 = np.meshgrid(w0_vals, w1_vals)
ax.plot_surface(B0, B1, Z, cmap='viridis')

#Addign labels
ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('Loss')
plt.show()
