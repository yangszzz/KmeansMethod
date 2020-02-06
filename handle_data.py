import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


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

    return AllAttacks, AllAttacks_labels

# use randomForestClassifier to reduct dimension
def count_dimension_value(AllAttacks, AllAttacks_labels):
    # RF = RandomForestRegressor(random_state=0, max_depth=10)
    #
    # RFmodel = RF.fit(AllAttacks, AllAttacks_labels)
    # feature_importances = RFmodel.feature_importances_
    RF = RandomForestClassifier(random_state=0, n_jobs=-1)
    RFmodel = RF.fit(AllAttacks.values, AllAttacks_labels.values.ravel())
    feature_importances = RFmodel.feature_importances_
    print(feature_importances)
    return feature_importances

#print the graph of the feature values
def print_histogram(x_list, y_list):
    plt.bar(x_list, y_list, 0.3, color='salmon', label='each_dimension_value')
    plt.show()

#降维
def reduct_dimension( AllAttacks, feature_value, feature_value_threshold, feature_correlation_threshold):
    #根据特征值权重删去权重较低的
    del_list = []
    for i in range(feature_value.shape[0]):
        if feature_value[i] <= feature_value_threshold:
            del_list.append(i)
    print(AllAttacks.columns[del_list])
    AllAttacks.drop(AllAttacks.columns[del_list], axis=1, inplace=True)
    feature_value = np.delete(feature_value, del_list)

    print('删除特征值权重小于设定值之后的特征数', AllAttacks.shape[1])

    #delete feature whose relation more than feature_correlation_threshold
    # feature_corr = AllAttacks.corr()
    #
    # del_list = []
    # for i in range(feature_corr.shape[0]):
    #     for j in range(i, feature_corr.shape[1]):
    #         if feature_corr.iloc[i][j] >= feature_correlation_threshold:
    #             print(feature_corr.iloc[i][j])
    #             if feature_value[i] > feature_value[j]:
    #                 del_list.append(j)
    #             else:
    #                 del_list.append(i)
    #
    # del_list = list(set(del_list))
    # AllAttacks.drop(AllAttacks.columns[del_list], axis=1, inplace=True)
    # feature_value = np.delete(feature_value, del_list)
    # print('删除特征值关联大于设定值之后的特征数', AllAttacks.shape[1])

    return AllAttacks, feature_value

def data_normalization(AllAttacks):
    #Three naturalization methods,Assess its good or bad individually
    #standard method
    columns = AllAttacks.columns.tolist()
    for c in columns:
        d = AllAttacks[c]
        AllAttacks[c] = ((d-d.mean())/(d.std())).tolist()
    return AllAttacks

if __name__ == '__main__':
    att_url, attLabel_url = 'C:/Users/NeverMore/Desktop/IDS/data/total_data.csv', 'C:/Users/NeverMore/Desktop/IDS/data/total_data_label.csv'
    reduced_dimensional_data = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data.csv'
    reduced_dimensional_data_label = 'C:/Users/NeverMore/Desktop/IDS/data/reduced_dimensional_data_label.csv'

    AllAttacks, AllAttacks_labels = load_data(att_url, attLabel_url)

    dimensionValues = count_dimension_value(AllAttacks, AllAttacks_labels)
    AllAttacks, dimensionValues = reduct_dimension(AllAttacks, dimensionValues, 0.01, 0.95)
    AllAttacks = data_normalization(AllAttacks)
    AllAttacks.to_csv(reduced_dimensional_data, index=None)
    AllAttacks_labels.to_csv(reduced_dimensional_data_label, index=None)