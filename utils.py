import numpy as np
import os
import math
from scipy.stats import norm
from tslearn.barycenters import softdtw_barycenter
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
from tslearn.metrics import soft_dtw


def get_labels(y):
    labels_set = set()
    labels_list = []
    for i in y:
        labels_set.add(i)
    for i in labels_set:
        labels_list.append(i)
    labels = np.asarray(labels_list, dtype=np.int)
    labels.shape = len(labels_set)
    return labels


def get_num_in_each_label(ytrain, labels):
    ans = np.zeros(shape=labels.size)
    for i in range(ytrain.size):
        for j in range(labels.size):
            if (ytrain[i] == labels[j]):
                ans[j] += 1
                break
    return ans


def barrycenters(xtrain, ytrain, labels, num_in_each_label):
    split = []
    barrycenterss = []
    barrycenters_all = []
    for j in range(labels.size):
        split.clear()
        barrycenterss.clear()
        for i in range(xtrain.shape[0]):
            if (ytrain[i] == labels[j]):
                split.append(xtrain[i])
        split_array = np.asarray(split, dtype=np.float64)
        # split_array.shape = (num_in_each_label[j], xtrain.shape[1])
        barrycenter = softdtw_barycenter(split_array, gamma=1.0, weights=None, method='L-BFGS-B', tol=0.001,
                                         max_iter=50,
                                         init=None)
        barrycenterss.append(barrycenter)
        barrycenters_all += barrycenterss
    barrycenters_all_array = np.asarray(barrycenters_all, dtype=np.float64)
    barrycenters_all_array.shape = (labels.size, xtrain.shape[1])
    return barrycenters_all_array


def draw(xtrain, ytrain, labels, barrycenters, path):
    N = labels.size
    n = N // 2
    m = N - n
    if (N == 2):
        n = N
        m = 0
    elif (N % 2 == 1):
        if (n < m):
            n, m = m, n
    f, sub = plt.subplots(nrows=2, ncols=n, squeeze=False, sharex=True, sharey=True,figsize=(15,15))
    for i in range(n):
        for j in range(ytrain.size):
            if (ytrain[j] == labels[i]):
                sub[0][i].plot(xtrain[j], c="blue")
        sub[0][i].plot(barrycenters[i], c="red",lw=5)
        str = f"Class {labels[i]}"
        sub[0][i].set_xlabel(str)
    for i in range(m):
        for j in range(ytrain.size):
            if (ytrain[j] == labels[i + n]):
                sub[1][i].plot(xtrain[j], c="blue")
        sub[1][i].plot(barrycenters[i + n], c="red",lw=5)
        str = f"Class {labels[i + n]}"
        sub[1][i].set_xlabel(str)
    f.savefig(path)


def load_data(file_name):
    folder_path = "/home/hadi/data sets/UCRArchive_2018/"
    folder_path += (file_name + "/")
    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"
    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None
    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)
    ytrain = train[:, 0]
    ytest = test[:, 0]
    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)
    return xtrain, ytrain, xtest, ytest


def normalisation(xtrain, xtest):
    xtrain = (xtrain - xtrain.mean(axis=1, keepdims=True)) / (xtrain.std(axis=1, keepdims=True))
    xtest = (xtest - xtest.mean(axis=1, keepdims=True)) / (xtest.std(axis=1, keepdims=True))
    return xtrain, xtest
