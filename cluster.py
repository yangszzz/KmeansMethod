import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics


def load_data(data_url, data_label_url):
    AllAttacks = pd.read_csv(data_url, low_memory=False)
    AllAttacks_labels = pd.read_csv(data_label_url, low_memory=False)
    return AllAttacks, AllAttacks_labels

#划分训练和测试集
def data2train_test(AllAttacks, AllAttacks_labels):
    AllAttacks = AllAttacks.values
    AllAttacks_labels = AllAttacks_labels.values
    train_X, test_X, train_y, test_y = train_test_split(AllAttacks,
                                                        AllAttacks_labels,
                                                        test_size=0.4,
                                                        random_state=0)

    return train_X, test_X, train_y, test_y


def cluster_data(train_data, test_data, train_data_labels, test_data_labels ):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=15, batch_size=30,
                      n_init=10)
    mbk.fit(train_data)
    prediction_labels = mbk.predict(test_data)
    result = metrics.v_measure_score(test_data_labels.ravel(), prediction_labels)
    return result


if __name__ == '__main__':
    data_url = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data.csv'
    data_label_url = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data_label.csv'
    AllAttacks, AllAttacks_labels = load_data(data_url, data_label_url)
    train_data, test_data, train_data_labels, test_data_labels = data2train_test(AllAttacks, AllAttacks_labels)
    result = cluster_data(train_data, test_data, train_data_labels, test_data_labels)
    print(result)