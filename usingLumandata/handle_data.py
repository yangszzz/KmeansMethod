# handle data

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


# reduct dimensions(using randomforest)
def reduct_dimension(data, data_labels, feature_value_threshold):
    # according to the threshold to reduct dimensions
    print('before reduct:', data.shape)
    RF = RandomForestClassifier(random_state=0, n_jobs=-1)
    RF.fit(data, data_labels.ravel())
    model = SelectFromModel(RF, prefit=True, threshold=feature_value_threshold)
    data = model.transform(data)
    print('after reduct:', data.shape)
    return data


# Split dataset into train dataset and test dataset
def data2train_test(data, data_labels):
    # Replace all attack type tags with 1,Replace all attack type tags with 1

    #     data_labels[data_labels > 0] = 1
    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        data_labels,
                                                        test_size=0.3,
                                                        random_state=0)
    # print(train_y.shape)
    return train_X, test_X, train_y, test_y


# test print settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    data1 = pd.read_csv('./data/train_data.csv', low_memory=False)
    data2 = pd.read_csv('./data/test_data.csv', low_memory=False)

    data = pd.concat([data1, data2])
    print(data.shape)
    data_labels = data['82']
    print(data_labels.shape)
    data = data.drop(columns=['81', '82'])
    print(data.shape)
    data = reduct_dimension(data, data_labels, 0.001)
    train_X, test_X, train_y, test_y = data2train_test(data, data_labels)
    pd.DataFrame(train_X).to_csv('./data/train_X.csv', index=None)
    pd.DataFrame(train_y).to_csv('./data/train_y.csv', index=None)
    pd.DataFrame(test_X).to_csv('./data/test_X.csv', index=None)
    pd.DataFrame(test_y).to_csv('./data/test_y.csv', index=None)