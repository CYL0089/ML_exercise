import numpy as np
import matplotlib.pyplot as plt

def loss(W, X, Y, m):
    """ w是权重向量,X是数据集, Xi是第i 个样本, m 是样本量 """
    loss_sum = 0
    for i in range(m):
        Xi = X[i]
        # print('--------------------------------')
        # print(Xi)
        # print(Xi.shape)
        # print(W.T)
        # print(W.T.shape)
        # print('--------------------------------')
        Yi = Y[i]
        hw_Xi = np.dot(W.T, Xi)
        loss_i = (hw_Xi - Yi) ** 2
        loss_sum += loss_i

    cost = loss_sum / (2 * m)

    return cost

def Grad_Descent(W, X, Y, m, n, lr):
    """ w是权重向量,X是数据集, Xi是第i 个样本, m 是样本量, n是特征个数 """
    for j in range(n):
        var = 0
        for i in range(m):
            Xi = X[i]
            Yi = Y[i]
            hw_Xi = np.dot(W.T, Xi)
            var += (hw_Xi - Yi) * X[i,j]
        dWj = (1 / m) * var
        W[j] = dWj - lr * dWj

    return W

if __name__ == '__main__':
    """房价预测：Xi = [1, 面积, 公共面积，房间数，家具个数]"""
    X = np.array([
         [1, 1200, 350, 3, 15],
         [1, 1500, 500, 4, 18],
         [1, 1000, 300, 3, 10],
         [1, 1800, 400, 5, 12],
         [1, 1000, 200, 2, 10],
         [1, 900, 300, 3, 10],
         [1, 800, 100, 2, 8],
         [1, 830, 200, 2, 9],
         [1, 900, 200, 2, 10],
         [1, 750, 150, 2, 10],
         [1, 950, 150, 3, 11]],dtype=np.float)
    """均值归一化"""
    X[:, 1] = (X[:, 1] - 1000) / 1050
    X[:, 2] = (X[:, 2] - 250) / 400
    X[:, 3] = (X[:, 3] - 2.5) / 3
    X[:, 4] = (X[:, 4] - 10) / 10
    # print(X)
    Y = np.array([3, 4, 2.5, 6, 2.1, 2.3, 1.8, 1.9, 2, 1.5, 2.1]) # 房价，单位：百万
    Y = Y / 10 # 当使用线性回归训练模型使用线性的数据(Y和X变化幅度一致)时，训练出的结果会更准确
    #     print(Y)
    W = np.array([0, 0.45, 0.25, 0.20, 0.1])

    lr = 0.001
    m = X.shape[0]
    n = W.shape[0]

    for i in range(1000):
        cost = loss(W, X, Y, m)
        W_new = Grad_Descent(W, X, Y, m, n, lr)
        print(W_new)
        if i%10 == 0:
            print('iter = {0}, loss = {1}'.format(i, cost))