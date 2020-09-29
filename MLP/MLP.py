import numpy as np
from sklearn.metrics import accuracy_score
import operator


class MLP:
    def __init__(self):
        self.input_size = 4
        self.hidden_size = 3
        self.output_size = 3
        self.hist = {'loss': [], 'acc': []}

        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.output_size)

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e, axis=1, keepdims=True)

    def sigmoid(self, x):
        result = np.zeros(x.shape)
        for i in range(x.shape[0]):
          for j in range(x.shape[1]):
            if x[i][j] < 0:
                a = np.exp(x[i][j])
                result[i][j]= a / (1 + a)
            else:
                result[i][j]= 1 / (1 + np.exp(-x[i][j]))
          return(result)

    def cross_entropy(self, y_in, o):
        t = y_in * np.log(o + 1e-10)
        return -np.sum(t)

    def forward(self, x):
        self.i = x
        self.z1 = np.dot(self.i, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.z2, self.a2

    def backward(self, Y):
        for y in range(len(Y)):
            self.a2[y,Y] = self.a2[y,Y]- 1
        self.a2 /= len(Y)
        dw2 = np.dot(np.transpose(self.a1),self.a2)
        d2b = np.sum(self.a2,axis=0)
        dw1 = np.dot(self.a2,np.transpose(self.W2))
        s = dw1 * (self.a1*(1-self.a1))
        dw1 = np.dot(np.transpose(self.i),s)
        d1b = np.sum(s,axis=0)

        self.W1 = self.W1 - 0.01 * dw1
        self.b1 = self.b1 - 0.01 * d1b
        self.W2 = self.W2 -  0.01 * dw2
        self.b2 = self.b2 -  0.01 * d2b
        return

    def train(self, x, y, epochs):
        for epoch in range(1, epochs + 1):
            y_in = [[0 for i in range(3)] for j in range(len(y))]
            for i in range(len(y)):
                if y[i] == 0:
                    y_in[i][0] = 1
                elif y[i] == 1:
                    y_in[i][1] = 1
                elif y[i] == 2:
                    y_in[i][2] = 1
            o,g = self.forward(x)
            self.loss = self.cross_entropy(y_in, g)
            self.backward(y)
            acc_o = np.argmax(o, axis=1)
            print(y)
            print(acc_o)
            acc = accuracy_score(y, acc_o)
            self.hist['loss'] += [self.loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', self.loss, 'acc:', acc)




