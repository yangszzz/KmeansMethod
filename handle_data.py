import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing


# load_data from csvfile
def load_data(att_url, attLabel_url):
    AllAttacks = pd.read_csv(att_url, low_memory=False)
    AllAttacks_labels = pd.read_csv(attLabel_url, low_memory=False)
    print(AllAttacks_labels[' Label'].value_counts())
    print(AllAttacks.shape[0])
    print(AllAttacks.shape[1])
    # AllAttacks_labels = AllAttacks[' Label']
    # AllAttacks = AllAttacks.drop(columns=[' Label'])
    AllAttacks_labels = AllAttacks_labels.replace(['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'DoS GoldenEye', 'FTP-Patator', 'SSH-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'Bot', 'Web Attack 锟?Brute Force', 'Web Attack 锟?XSS', 'Infiltration', 'Web Attack 锟?Sql Injection', 'Heartbleed'],\
            [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])

    # data, data_label(array)
    return AllAttacks.values, AllAttacks_labels.values

# use randomForestClassifier to reduct dimension
def count_dimension_value(AllAttacks, AllAttacks_labels):
    # RF = RandomForestRegressor(random_state=0, max_depth=10)

    # RFmodel = RF.fit(AllAttacks, AllAttacks_labels)
    # feature_importances = RFmodel.feature_importances_
    RF = RandomForestClassifier(random_state=0, n_jobs=-1)
    RFmodel = RF.fit(AllAttacks, AllAttacks_labels.ravel())
    feature_importances = RFmodel.feature_importances_
    print(feature_importances)
    return feature_importances

#print the graph of the feature values
def print_histogram(x_list, y_list):
    plt.bar(x_list, y_list, 0.3, color='salmon', label='each_dimension_value')
    plt.show()

#降维
def reduct_dimension( AllAttacks, feature_value, feature_value_threshold):
    #根据特征值权重删去权重较低的
    print(feature_value.shape)
    del_list = []
    for i in range(feature_value.shape[0]):
        if feature_value[i] <= feature_value_threshold:
            del_list.append(i)
    # print(AllAttacks.columns[del_list])
    AllAttacks = np.delete(AllAttacks, del_list, axis=1)
    feature_value = np.delete(feature_value, del_list)

    print('删除特征值权重小于设定值之后的特征数', AllAttacks.shape)

    return AllAttacks, feature_value

#sigmoid function to normalization
def sigmoid_func(x):
    return 1.0/(1+np.exp(-x))

def data_normalization(AllAttacks):
    # naturalization methods
    minMaxScaler = preprocessing.MinMaxScaler()
    minMax = minMaxScaler.fit(AllAttacks)
    AllAttacks = minMax.transform(AllAttacks)
    return AllAttacks


if __name__ == '__main__':
    att_url, attLabel_url = 'C:/Users/NeverMore/Desktop/IDS/data/total_data.csv', 'C:/Users/NeverMore/Desktop/IDS/data/total_data_label.csv'
    reduced_dimensional_data = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data.csv'
    reduced_dimensional_data_label = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data_label.csv'

    AllAttacks, AllAttacks_labels = load_data(att_url, attLabel_url)

    dimensionValues = count_dimension_value(AllAttacks, AllAttacks_labels)
    AllAttacks, dimensionValues = reduct_dimension(AllAttacks, dimensionValues, 0.01)
    AllAttacks = data_normalization(AllAttacks)

    AllAttacks = pd.DataFrame(AllAttacks)
    AllAttacks_labels = pd.DataFrame(AllAttacks_labels)
    #save to csv.file
    AllAttacks.to_csv(reduced_dimensional_data, index=None)
    AllAttacks_labels.to_csv(reduced_dimensional_data_label, index=None)