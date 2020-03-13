import pandas as pd
import numpy as np
import os
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


#test print settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


down_dic = {
    'BENIGN':                        250000,
    'DoS Hulk':                      230124,
    'PortScan':                      158804,
    'DDoS':                          128025,
    'DoS GoldenEye':                  10293,
    'FTP-Patator':                     7935,
    'SSH-Patator':                  5897,
    'DoS slowloris':                5796,
    'DoS Slowhttptest':              5499,
    'Bot':                        1956,
    'Web Attack 锟?Brute Force':          1507,
    'Web Attack 锟?XSS':                  652,
    'Infiltration':                     36,
    'Web Attack 锟?Sql Injection':         21,
    'Heartbleed':                      11,
}
up_dic = {
    'BENIGN': 250000,
    'DoS Hulk': 230124,
    'PortScan': 158804,
    'DDoS': 128025,
    'DoS GoldenEye': 10293,
    'FTP-Patator': 7935,
    'SSH-Patator': 5897,
    'DoS slowloris': 5796,
    'DoS Slowhttptest': 5499,
    'Bot': 5000,
    'Web Attack 锟?Brute Force': 5000,
    'Web Attack 锟?XSS': 5000,
    'Infiltration': 5000,
    'Web Attack 锟?Sql Injection': 5000,
    'Heartbleed': 5000,
}

data_url = 'C:/Users/NeverMore/Desktop/IDS/data/complete_data/MachineLearningCVE'
processed_data_url = 'C:/Users/NeverMore/Desktop/IDS/data/total_data.csv'
processed_label_url = 'C:/Users/NeverMore/Desktop/IDS/data/total_data_label.csv'


file_list = os.listdir(data_url)

aa = []
ll = []
for i in range(len(file_list)):
    item = pd.read_csv(data_url+'/'+file_list[i], low_memory=False)
    item = item[~item.isin(['Infinity'])]
    item = item.dropna(axis=0, how='any')
    ll.append(item[' Label'])
    item = item.drop(columns=[' Label'])
    aa.append(item)

AllAttacks = pd.concat(aa, ignore_index=True)
AllAttacks_labels = pd.concat(ll, ignore_index=True)

AllAttacks = AllAttacks.loc[:, ~(AllAttacks==0).all()]#删除全零的列


#downsamping
DownSamp = RandomUnderSampler(sampling_strategy=down_dic, random_state=0)
AllAttacks, AllAttacks_labels = DownSamp.fit_sample(AllAttacks, AllAttacks_labels)
#oversamping
OverSamp = RandomOverSampler(sampling_strategy=up_dic, random_state=0)
AllAttacks, AllAttacks_labels = OverSamp.fit_sample(AllAttacks, AllAttacks_labels)
AllAttacks_labels = AllAttacks_labels.to_frame()

AllAttacks.to_csv(processed_data_url, index=None)
AllAttacks_labels.to_csv(processed_label_url, index = None)