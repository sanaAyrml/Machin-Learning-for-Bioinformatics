import numpy as np
import math

class NodeDT(object):
    def __init__(self, parent, depth):
        self.label = -1
        self.childeren = dict()
        self.parent = parent
        self.depth = depth
        self.predict = None
        self.p0 = 0
        self.p1 = 0

    def append_child(self, child, i):
        self.childeren[i] = child

    def make_leaf(self):
        self.is_leaf = True


class DecisionTree(object):
    def __init__(self, max_depth, threshhold):
        self.max_depth = max_depth
        self.threshhold = threshhold
        self.root_node = None

    def Information_gain(self, X, Y, xnum):
        ys = dict()
        for i in range(len(Y)):
            if not ys.__contains__(Y[i]):
                ys[Y[i]] = []
            ys[Y[i]].append(i)
        sum_all = 0
        for i in ys.keys():
            sum_all += len(ys[i])
        yp = dict()
        for i in ys.keys():
            yp[i] = len(ys[i]) / sum_all
        H = 0
        for i in ys.keys():
            H -= yp[i] * math.log2(yp[i])
        xs = dict()
        for i in range(len(X.transpose()[xnum])):
            if not xs.__contains__(X.transpose()[xnum][i]):
                xs[X.transpose()[xnum][i]] = []
            xs[X.transpose()[xnum][i]].append(i)
        H2 = 0
        for x in xs.keys():
            ys = dict()
            yp = dict()
            # print(Y[xs[x]])
            for y in range(len(Y[xs[x]])):
                if not ys.__contains__(Y[xs[x]][y]):
                    ys[Y[xs[x]][y]] = []
                ys[Y[xs[x]][y]].append(y)
            sum_all = 0
            for i in ys.keys():
                sum_all += len(ys[i])
            for i in ys.keys():
                yp[i] = len(ys[i]) / sum_all
            for i in ys.keys():
                H2 -= len(xs[x]) / len(Y) * yp[i] * math.log2(yp[i])
        return xs,H2,H

    def make_node(self, X, Y, node):
        ys = dict()
        for i in range(len(Y)):
            if not ys.__contains__(Y[i]):
                ys[Y[i]] = []
            ys[Y[i]].append(i)
        # print(len(ys.keys()))
        if len(ys.keys()) == 1:
            for i in ys.keys():
                # print("here",i)
                node.predict = i
                if i == 0:
                    node.p0 = 1
                    node.p1 = 0
                else:
                    node.p0 = 0
                    node.p1 = 1
            return
        for i in ys.keys():
            if len(ys[i]) / len(Y) > self.threshhold:
                # print("here",i)
                node.predict = i
                if i == 0:
                    node.p0 = len(ys[i]) / len(Y)
                    node.p1 = 1 - len(ys[i]) / len(Y)
                else:
                    node.p0 = 1 - len(ys[i]) / len(Y)
                    node.p1 = len(ys[i]) / len(Y)
                return
        maxx = -math.inf
        for i in ys.keys():
            if len(ys[i]) > maxx:
                maxx = i
        # print("here",maxx)
        minii = -math.inf
        xssP, label = None, None
        if node.depth < self.max_depth:
            for i in range(len(X[0])):
                if i != node.label:
                    xss, H2,H = self.Information_gain(X, Y, i)
                    if H - H2 > minii:
                        xssP = xss
                        label = i
                        minii = H - H2
            # print(xssP)
            if minii > 0:
                node.label = label
                for i in xssP.keys():
                    new_node = NodeDT(node, node.depth + 1)
                    node.append_child(new_node, i)
                    self.make_node(X[xssP[i]], Y[xssP[i]], new_node)
            else:
                # print("here",maxx)
                if maxx == 0:
                    node.p0 = len(ys[maxx]) / len(Y)
                    node.p1 = 1 - len(ys[maxx]) / len(Y)
                else:
                    node.p0 = 1 - len(ys[maxx]) / len(Y)
                    node.p1 = len(ys[maxx]) / len(Y)
                node.predict = maxx
        else:
            # print("here",maxx)
            if maxx == 0:
                node.p0 = len(ys[maxx]) / len(Y)
                node.p1 = 1 - len(ys[maxx]) / len(Y)
            else:
                node.p0 = 1 - len(ys[maxx]) / len(Y)
                node.p1 = len(ys[maxx]) / len(Y)
            node.predict = maxx

    def fit(self, X, Y):
        node = NodeDT(None, 1)
        self.root_node = node
        self.make_node(X[np.arange(len(Y))], Y[np.arange(len(Y))], node)

    def predict(self, X):
        Yp = []
        for i in range(len(X)):
            node = self.root_node
            while node.label != -1:
                j = 0
                # if i != 9:
                while not (node.childeren.__contains__(X[i][node.label] + j) or node.childeren.__contains__(
                        X[i][node.label] - j)):
                    j += 1
                # else:
                #     while not (node.childeren.__contains__(X[i][node.label]+j) or node.childeren.__contains__(X[i][node.label]-j)):
                #         j += 0.1
                if node.childeren.__contains__(X[i][node.label] + j):
                    new_node = node.childeren[X[i][node.label] + j]
                else:
                    new_node = node.childeren[X[i][node.label] - j]
                node = new_node
            Yp.append(node.predict)
        return Yp

    def predict_prob(self, X):
        Yp = []
        for i in range(len(X)):
            node = self.root_node
            while node.label != -1:
                j = 0
                # if i != 9:
                while not (node.childeren.__contains__(X[i][node.label] + j) or node.childeren.__contains__(
                        X[i][node.label] - j)):
                    j += 1
                # else:
                #     while not (node.childeren.__contains__(X[i][node.label]+j) or node.childeren.__contains__(X[i][node.label]-j)):
                #         j += 0.1
                if node.childeren.__contains__(X[i][node.label] + j):
                    new_node = node.childeren[X[i][node.label] + j]
                else:
                    new_node = node.childeren[X[i][node.label] - j]
                node = new_node
            Yp.append(node.p1)
        return Yp
