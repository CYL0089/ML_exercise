import numpy as np
import matplotlib.pyplot as plt
def loss(w, b, x, y):
    y_hat = w*x + b
    cost = (1/2) * (y_hat - y) ** 2
    loss = np.mean(np.sum(cost))

    return loss

def gradient(x, y, w, b):
    w_grad = (w * x + b - y) * x
    w_grad = np.mean(np.sum(w_grad))
    b_grad = w * x + b - y
    b_grad = np.mean(np.sum(b_grad))
    w_new = w - lr * w_grad
    b_new = b - lr * b_grad
    return w_new, b_new

if __name__ == '__main__':
    w = 0
    b = 0
    data = np.array([[1,1], [2,3], [5, 4], [8, 10]])
    x = data[:,:1]
    y = data[:,1:]
    lr = 0.01

    for i in range(1000):
       w, b = gradient(x, y, w, b)
       cost = loss(w, b, x, y)
       if i%10 == 0:
           print('i = {0},  loss = {1}'.format(i, cost))
           print('w = {0},  b = {1}'.format(w, b))
    plt.scatter(x, y)
    plt.plot(x, x*w+b)
    plt.show()
