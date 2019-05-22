import numpy as np
from e_com_data_process import get_data
X, Y = get_data()
M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D,M)
b1 = np.zeros(M)
w2 = np.random.randn(M,K)
b2 = np.zeros(K)


def soft_max(a):
    expa = np.exp(a)
    return expa/expa.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z= np.tanh(X.dot(W1)+b1)
    return soft_max(Z.dot(W2)+b2)


P_Y_given_X = forward(X,W1,b1,w2,b2)
prediction = np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y,P):
    return np.mean(Y == P)

print(f'classification rate is {classification_rate(Y,prediction)}')


