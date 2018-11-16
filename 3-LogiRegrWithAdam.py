import numpy as np
import pandas as pd

# parameters: m样本数，n特征数，

def loaddata(filename):
    X = np.array([])
    Y = np.array([])
    for line in open(filename, 'r'):
        X = np.append(X, line[:-2])
        Y = np.append(Y, line[-1])

    return X, Y

def costFuntion(m, n, X, Y, W, b):
    # 初始化
    dW = np.zeros((n, 1))
    db = 0
    cost = 0
    Z = np.array([])
    a = np.array([])
    # for i in range(m):
    #     Z[i] = np.dot(W, X[i]) + b
    #     a[i] = sigmoid(Z[i])
    #     cost += -Y[i] * np
    Z = np.dot(W, X) + b
    A = sigmoid(Z)   # 相当于预测值 y_hat
    cost = (-Y * np.log(A) - (1 - Y) * np.log(1- A)) / m

    return cost

def Adam(iter, alpha, Vdw, Sdw, Vdb, Sdb, X, Y, Z, A, W, b, m, beita1=0.9, beita2=0.999, epsilon=(10 ** -8)):
    # dW = np.zeros((n, 1))
    # db = 0
    # cost = 0
    dZ = A - Y
    dW = np.dot(dZ, X) / m
    db = np.sum(dZ)

    # # 初始化
    # Vdw, Sdw, Vdb, Sdb = 0,0,0,0
    # ----------- Momentum ----------------
    Vdw = beita1 * Vdw + (1 - beita1) * dW
    Vdb = beita1 * Vdb + (1 - beita1) * db
    # ----------- RMSprop ----------------
    Sdw = beita2 * Sdw + (1 - beita2) * np.square(dW)
    Sdb = beita2 * Sdb + (1 - beita2) * np.square(db)
    # ----------- 偏差修正 ----------------
    Vdw_corrected = Vdw / (1 - beita1 ** iter)
    Vdb_corrected = Vdb / (1 - beita1 ** iter)
    Sdw_corrected = Sdw / (1 - beita2 ** iter)
    Sdb_corrected = Sdb / (1 - beita2 ** iter)
    # ----------- 更新参数 ----------------
    W = W - alpha * Vdw_corrected / (np.sqrt(Sdw_corrected) + epsilon)
    b = b - alpha * Vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon)

    return W, b

def sigmoid(Z):
    a = 1 / (1 + np.exp(-Z))

    return a

def train():
    return 0

if __name__ == '__main__':

    print()