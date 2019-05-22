import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from e_com_data_process import get_data


def y2_indicator(y,k):
    N = len(y)
    ind = np.zeros((N, k))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))
X_train = X[:-100]
Y_train = Y[:-100]

Ytrain_ind = y2_indicator(Y_train,K)
X_test = X[-100:]
Y_test = Y[-100:]
Ytest_ind = y2_indicator(Y_test,K)

W = np.random.randn(D, K)
b = np.zeros(K)


def softmax(a):
    expa = np.exp(a)
    return expa/expa.sum(axis=1, keepdims=True)


def forward(X, W, b):
    return softmax(X.dot(W)+b)


def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)


def classification_rate(Y, P):
    return np.mean(Y==P)


def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))


train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
    pYtrain = forward(X_train, W, b)
    pYtest = forward(X_test, W, b)
    ctrain = cross_entropy(Ytrain_ind,pYtrain)
    ctest = cross_entropy(Ytest_ind,pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W -= learning_rate*X_train.T.dot(pYtrain-Ytrain_ind)
    b -= learning_rate*(pYtrain - Ytrain_ind).sum(axis=0)

    if i%1000 == 0:
        print(f'{i} {ctrain} {ctest}')

print(f'final train classification rate: {classification_rate(Y_train,predict(pYtrain))}')
print(f'final test classification rate: {classification_rate(Y_test,predict(pYtest))}')

legend1 = plt.plot(train_costs, label='train cost')
legend2 = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()






