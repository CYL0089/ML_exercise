import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))

    return a

def initWeight(n):
    W = np.zeros(n)
    np.random.seed(n)
    for i in range(n):
        W[i] = np.random.random

    return W

def lossFunction(m, Y_hat, Y): # m为样本量，Y_hat为预测值，Y 为标签
    first = np.dot(Y, np.log(Y_hat))
    second = np.dot((1 - Y), np.log(1 - Y_hat))
    cost = (-1 / m ) * np.sum(first + second)

    return cost

def MiniBatch_GradientDescent(batch_size,X, Y, Y_hat, W, lr):
    dW = (1 / batch_size) * np.sum(np.dot((Y_hat - Y), X))
    W = W - lr * dW

    return W

def train(batch_size, X_batch, Y_batch, Y_hat_batch, m, W, lr):
    # 停止策略：迭代次数
    for iter in range(800):
        W = MiniBatch_GradientDescent(batch_size, X_batch, Y_batch, Y_hat_batch, W, lr)
        if iter % 50 == 0:
            cost = lossFunction(batch_size, Y_hat_batch, Y_batch)
            print('iter = {0}, cost = {1}'.format(iter, cost))

if __name__ == '__main__':
    data = pd.read_table('LogiReg_data.txt', sep = ',')
    X = data.iloc[:, 0:2]
    X[-1] = pd.Series(np.array([1] * 100)) # X 加一列作为偏置b
    Y = data.iloc[:, -1]

    # 权重的随机初始化：
    W = initWeight(X.shape[1])

    lr = 0.0002
    batch_size = 64
    # X_batch, Y_batch, Y_hat_batch
    # train(......)






