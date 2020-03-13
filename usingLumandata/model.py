# Parameter adjustment with k-fold cross validation and grid search

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import svm, metrics
from scipy.spatial.distance import cdist



def load_data(data_url, data_label_url):
    data = pd.read_csv(data_url, low_memory=False)
    data_labels = pd.read_csv(data_label_url, low_memory=False)

    return data.values, data_labels.values


# Spearate data:train data and test data
def data2train_test(AllAttacks, AllAttacks_labels):
    # Replace all attack type tags with 1,Replace all attack type tags with 1
    AllAttacks_labels[AllAttacks_labels > 0] = 1

    train_X, test_X, train_y, test_y = train_test_split(AllAttacks,
                                                        AllAttacks_labels,
                                                        test_size=0.4,
                                                        random_state=0)
    # print(train_y.shape)
    return train_X, test_X, train_y, test_y


# Separate data: normal data and abnormal data
def split_normal_attack(data, data_labels):
    normal_train = data[np.where(data_labels[:, -1] == 0)]
    abnormal_train = data[np.where(data_labels[:, -1] == 1)]
    normal_train_label = data_labels[np.where(data_labels[:, -1] == 0)]
    abnormal_train_label = data_labels[np.where(data_labels[:, -1] == 1)]
    return normal_train, normal_train_label, abnormal_train, abnormal_train_label


# cluster data and confirm the best clusters for this model
def cluster_data(data, n_clusters):
    # confirm the best cluster numbers
    k = n_clusters
    kl = []
    while k <= 40:
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=30, n_init=10, random_state=0)
        mbk.fit(train_data)
        # Used to determine inflection point to determine the number of clusters.
        #         kl.append(sum(np.min(cdist(train_data,mbk.cluster_centers_,'euclidean'),axis=1))/train_data.shape[0])
        kl.append(mbk.inertia_)
        k += 1
    plt.figure()
    plt.plot([i for i in range(n_clusters, k)], kl, 'bo-', mfc='r')


# return the clusters centers for the svm
def cluster_centers(data, data_label, n_clusters):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=30,
                          n_init=10)
    mbk.fit(data)
    centers_labels = np.array([[data_label] for i in range(n_clusters)])
    return mbk.cluster_centers_, centers_labels


def E(n):
    return 10 ** n


def svm_classify(data, data_labels, test, test_labels):
    parameters = {'C': [E(i) for i in range(-6, 10)], 'gamma': np.arange(0.01, 0.1, 0.01)}
    # kernel = 'rbf'
    for i in range(3, 11):
        kfold = StratifiedKFold(n_splits=i, random_state=0, shuffle=False)
        clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=kfold, scoring='accuracy')
        clf.fit(data, data_labels.ravel())
        print('The optimal parameters：', clf.best_params_)
        print('Highest validation set score：', clf.best_score_)
        # 获取最优模型
        best_model = clf.best_estimator_
        print('Accuracy on the test set：', best_model.score(test, test_labels))
        print('k,n_normal:', i, len(data_labels))
        print()


def show_accuracy(a, b):
    # 计算预测值和真实值一样的正确率
    acc = a.ravel() == b.ravel()
    #     print('precision:%.5f%%' % ((100*float(acc.sum()))/a.size))
    return (float(acc.sum())) / a.size


if __name__ == '__main__':
    train_data_url = './data/train_X.csv'
    train_data_labels_url = './data/train_y.csv'
    test_data_url = './data/test_X.csv'
    test_data_labels_url = './data/test_y.csv'

    train_data, train_data_labels = load_data(train_data_url, train_data_labels_url)
    test_data, test_data_labels = load_data(test_data_url, test_data_labels_url)

    # split the train data into normal and abnormal
    normal_train, normal_train_label, abnormal_train, abnormal_train_label = split_normal_attack(train_data,
                                                                                                 train_data_labels)

    # The following two lines of code run to see the results to determine the optimal number of clusters
    cluster_data(normal_train, 10)
    cluster_data(abnormal_train, 14)
    # result:    normal(3) abnormal(15，20)

    # After the last two lines of code are run, select the best number of clusters according to the results to run the following code
    for j in range(8,17):
        normal_centers,normal_centers_labels = cluster_centers(normal_train,0,20)
        abnormal_centers,abnormal_centers_labels = cluster_centers(abnormal_train,1,15)
        train = np.vstack((normal_centers,abnormal_centers))
        train_labels = np.vstack((normal_centers_labels,abnormal_centers_labels))

        svm_classify(train,train_labels,test_data,test_data_labels)