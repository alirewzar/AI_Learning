import numpy as np
import matplotlib.pyplot as plt

def J_convex(w):
    return w[0]**2 + 2 * w[1]**2

def grad_J_convex(w):
    return np.array([2 * w[0] , 4 * w[1]])



def gradient_descent(grad, w_init, learning_rate, n_steps):
    w =  np.array(w_init)
    path = [w.copy()]
 
    for step in range(n_steps):
        grad_w = grad(w)
        w = w - learning_rate * grad_w
        path.append(w.copy())

    return np.array(path)
    
w_init = [4, 4]
path_convex = gradient_descent(grad_J_convex, w_init, learning_rate=0.1, n_steps=50)

w1_vals = np.linspace(-5, 5, 400)
w2_vals = np.linspace(-5, 5, 400)
W1,W2 = np.meshgrid(w1_vals, w2_vals)
J_vals = J_convex([W1, W2])

plt.figure(figsize=(8, 6))
contour_levels = np.logspace(-0.5, 3, 35)
plt.contour(W1, W2, J_vals, levels=contour_levels, camp="jet")
path_x, path_y = path_convex[:, 0], path_convex[:, 1]
plt.plot(path_x, path_y, marker='o', color='red', label = 'Gradient Descent path')
plt.title("Gradient Descent on a Convex Function")
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend()
plt.show()