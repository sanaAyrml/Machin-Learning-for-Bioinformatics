import pandas as pd
import numpy as np
import random
import math
from collections import OrderedDict
import matplotlib.pyplot as plt

def read_data(address, column_name):
    data = pd.read_csv(address, delimiter=",")
    print(data)
    Y = data[column_name].values
    data = data.drop([column_name], axis=1)
    X = data.values
    return X, Y


def preprocessDT(X, Y):
    for i in range(len(X)):
        X[i][0] = int(X[i][0] / 10)
        X[i][3] = int(X[i][3] / 10)
        X[i][4] = int(X[i][4] / 20)
        X[i][9] = int((X[i][9] * 10) / 10)
        X[i][7] = int(X[i][7] / 10)
    return X, Y


def shuffle(X, Y):
    s = np.arange(len(Y))
    random.shuffle(s)
    return X[s], Y[s]


def split(X, Y, rate):
    s = np.arange(len(Y))
    b = int(len(Y) * (1 - rate))
    strain = s[0:b]
    stest = s[b:len(Y)]
    return X[strain], Y[strain], X[stest], Y[stest]

def find_percision(predicted, real):
    percision = 0
    for i in range(len(predicted)):
        if predicted[i] == real[i]:
            percision += 1
    return percision / len(predicted)

def fivefold(X, Y):
    XsTrain = []
    YsTrain = []
    XsTest = []
    YsTest = []
    s = np.arange(len(Y))
    b1 = int(len(Y)/5)
    parts= []
    for i in range(5):
        p = np.arange(i*b1,i*b1+b1)
        parts.append(p)
    for i in parts:
        XsTest.append(X[i])
        YsTest.append(Y[i])
        XsTrain.append(np.concatenate((X[0:i[0]], X[i[len(i) - 1]+1:len(X)])))
        YsTrain.append(np.concatenate((Y[0:i[0]], Y[i[len(i) - 1]+1:len(Y)])))

    return XsTrain,YsTrain, XsTest, YsTest

def find_percision(predicted, real):
    percision = 0
    for i in range(len(predicted)):
        if predicted[i] == real[i]:
            percision += 1
    return percision / len(predicted)


def confusion_matrix(predicted, real):
    labels = dict()
    n = 0
    for i in predicted:
        if not labels.__contains__(i):
            labels[i] = n
            n += 1
    for i in real:
        if not labels.__contains__(i):
            labels[i] = n
            n += 1
    cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    # tn, fp, fn, tp
    for i in range(len(predicted)):
        cm[labels[real[i]]][labels[predicted[i]]] += 1
    print("Confusion Matrix: ", cm)
    return cm


def classification_report(predicted, real):
    cm = confusion_matrix(predicted, real)
    Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + (cm[1][0] + cm[0][1]))
    Precision = cm[1][1] / (cm[1][1] + cm[0][1])
    Recall = cm[1][1] / (cm[1][1] + cm[1][0])
    F1Score = 2 * (Recall * Precision) / (Recall + Precision)
    Specificity = cm[0][0] / (cm[0][0] + cm[0][1])
    print("Accuracy: ", Accuracy)
    print("Precision: ", Precision)
    print("Recall: ", Recall)
    print("F1Score: ", F1Score)
    print("Specificity: ", Specificity)
    return Accuracy, Precision, Recall, F1Score, Specificity

def draw_ROC(y_scores,Ytest):
    fpr = []
    tpr = []
    thresholds = np.arange(0.0, 1.001, 0.001)
    print(len(thresholds))
    P = sum(Ytest)
    N = len(Ytest) - P
    for thresh in thresholds:
        FP = 0
        TP = 0
        for i in range(len(y_scores)):
            if (y_scores[i] > thresh):
                if Ytest[i] == 1:
                    TP = TP + 1
                if Ytest[i] == 0:
                    FP = FP + 1
        fpr.append(FP / float(N))
        tpr.append(TP / float(P))

    plt.plot(fpr, tpr, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    return

def t_test(a,b):
    a_mean = sum(a)/len(a)
    b_mean = sum(b)/len(b)
    d = []
    for i in range(len(a)):
        d.append(math.pow((a[i]-a_mean)-(b[i]-b_mean),2))
    t = (a_mean-b_mean)*math.sqrt((len(a)*(len(a)-1))/sum(d))
    s = np.random.standard_t(len(d), size=100000)
    p = np.sum(s<t) / float(len(s))
    print(p)
    # using a two-sided test
    print("P-value is: {}  ".format(2 * min(p, 1 - p) * 100))
    return t
