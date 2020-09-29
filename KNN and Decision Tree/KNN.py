import math
from collections import OrderedDict

class NodeKNN(object):
    def __init__(self, features, label):
        self.label = label
        self.features = features


class kNN(object):
    def __init__(self, K):
        self.K = K
        self.Nodes = []

    def fit(self, X, Y):
        for i in range(len(Y)):
            node = NodeKNN(X[i], Y[i])
            self.Nodes.append(node)

    def cal_dist(self, node1, node2):
        sum = 0
        for i in range(len(node1)):
            sum += pow((node1[i] - node2[i]), 2)
        return math.sqrt(sum)

    def predict(self, X):
        Yp = []
        for i in range(len(X)):
            distances = []
            for j in range(len(self.Nodes)):
                distances.append([self.Nodes[j].label, self.cal_dist(self.Nodes[j].features, X[i])])
            distances.sort(key=lambda distances: distances[1])
            nn = dict()
            for j in range(self.K):
                if not nn.__contains__(distances[j][0]):
                    nn[distances[j][0]] = 0
                nn[distances[j][0]] += 1
            nn_sorted = OrderedDict(sorted(nn.items(), key=lambda x: -x[1]))
            for k, v in nn_sorted.items():
                Yp.append(k)
                break
        return Yp

    def predict_prob(self, X):
        Yp = []
        for i in range(len(X)):
            distances = []
            for j in range(len(self.Nodes)):
                distances.append([self.Nodes[j].label, self.cal_dist(self.Nodes[j].features, X[i])])
            distances.sort(key=lambda distances: distances[1])
            nn = dict()
            nn[0] = 0
            nn[1] = 0
            for j in range(self.K):
                nn[distances[j][0]] += 1
            Yp.append( nn[1] / (nn[0] + nn[1]))
        return Yp
