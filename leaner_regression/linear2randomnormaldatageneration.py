import numpy as np
import matplotlib.pyplot as plt

n = 3000

""" 
if you want to know about how this row works you should 
know about the variance and the standard deviation 

becouse when you run this:
np.random.normal(0, 5, size=n)

your output datas mean is 0 and its standard deviation is 5

Population Variance (σ²):
σ² = (1 / N) * Σ (xᵢ - μ)²
Where:
σ² = Population variance
N = Total number of data points
xᵢ = Each individual data point
μ = Population mean (μ = (1 / N) * Σ xᵢ)

Sample Variance (s²):
s² = (1 / (n - 1)) * Σ (xᵢ - x̄)²
Where:
s² = Sample variance
n = Total number of data points in the sample
xᵢ = Each individual data point in the sample
x̄ = Sample mean (x̄ = (1 / n) * Σ xᵢ)

Population Standard Deviation (σ):
σ = √σ² = √((1 / N) * Σ (xᵢ - μ)²)

Sample Standard Deviation (s):
s = √s² = √((1 / (n - 1)) * Σ (xᵢ - x̄)²)
"""

x = np.random.normal(0, 10, size=n)
y =  -15 * x + 20 + np.random.normal(0, 10, size=n)


X = np.sum(x)
Y = np.sum(y)

w1 = (n * np.dot(x, y) - Y * X) / (n * np.dot(x, x) - X ** 2)
w0 = (Y - w1 * X) / n
w1, w0 = round(w1, 2), round(w0, 2)
print(f"y  = {w1} x + {w0}")

# Define the equation y = -15x + 20
yexact = w1 * x + w0

plt.figure(figsize=(12, 8))  # Set figure size

plt.plot(x, y, label="sampels", color="blue")

plt.plot(x, yexact, label="y = w1x + w0", color="red")


# Add labels and title
plt.xlabel("x")
plt.ylabel("y")

plt.title("samples vs prediction")

# Add grid and legend
plt.grid(True)
plt.legend()

plt.show()