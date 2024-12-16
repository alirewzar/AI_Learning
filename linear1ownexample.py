n = 5
x = [5, 3, -1, 2, 6]
y = [14, 6, -5.5, 3.5, 18]


X = sum(x)
Y = sum(y)

#for xiyi = sum(x(i)*y(i)) for i 0 to n.I want to calculate this with:
xiyi = 0
for i in range(n):
    xiyi += x[i] * y[i]

#for x2 = sum(x(i)^2)
xi2 = 0
for i in range(n):
    xi2 += pow(x[i],2)

w1 = (n * xiyi - X * Y)/(n * xi2 - pow(X,2))
w0 = (Y - w1 * X) / n

print(w0, w1)