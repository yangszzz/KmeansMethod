import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from scipy.spatial.distance import cdist



def load_data(data_url, data_label_url):
    AllAttacks = pd.read_csv(data_url, low_memory=False)
    AllAttacks_labels = pd.read_csv(data_label_url, low_memory=False)
    return AllAttacks.values, AllAttacks_labels.values


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
    while k <= 20:
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=k, batch_size=30,
                              n_init=10)
        mbk.fit(train_data)
        # Used to determine inflection point to determine the number of clusters.
        kl.append(sum(np.min(cdist(train_data, mbk.cluster_centers_, 'euclidean'), axis=1)) / train_data.shape[0])
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


def svm_classify(data, data_labels, test, test_labels):
    clf = svm.SVC()
    clf.fit(data, data_labels)
    pred = clf.predict(test)
    show_accuracy(pred, test_labels)


def show_accuracy(a, b):
    # 计算预测值和真实值一样的正确率
    acc = a.ravel() == b.ravel()
    print('precision:%.2f%%' % ((100 * float(acc.sum())) / a.size))


if __name__ == '__main__':
    data_url = './data/reduced_dimensional_data.csv'
    data_label_url = './data/reduced_dimensional_data_label.csv'
    AllAttacks, AllAttacks_labels = load_data(data_url, data_label_url)

    train_data, test_data, train_data_labels, test_data_labels = data2train_test(AllAttacks, AllAttacks_labels)
    # split the train data into normal and abnormal
    normal_train, normal_train_label, abnormal_train, abnormal_train_label = split_normal_attack(train_data,
                                                                                                 train_data_labels)

    # The following two lines of code run to see the results to determine the optimal number of clusters
    #     cluster_data(normal_train,1)
    #     cluster_data(abnormal_train,14)

    # After the last two lines of code are run, select the best number of clusters according to the results to run the following code
    normal_centers, normal_centers_labels = cluster_centers(normal_train, 0, 6)
    abnormal_centers, abnormal_centers_labels = cluster_centers(abnormal_train, 1, 18)
    train = np.vstack((normal_centers, abnormal_centers))
    train_labels = np.vstack((normal_centers_labels, abnormal_centers_labels))

    svm_classify(train, train_labels, test_data, test_data_labels)