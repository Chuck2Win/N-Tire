# -*- coding: utf-8 -*-
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder
from pandas import DataFrame as df
import numpy as np
import pickle
parser = argparse.ArgumentParser(description = 'data set')
parser.add_argument('--data', type = str , default = 'G:/넥센타이어/Foot Print/footprint_v2.csv' , help = '데이터 저장 위치')
parser.add_argument('--output_data', type = str , default = 'G:/넥센타이어/Foot Print/preprocessed_data.csv' , help = '데이터 저장 위치')
class dataset:
    def __init__(self):
        self.data = pd.read_csv(args.data,header=0)
        self.data['Data57']=self.data['Data57'].astype('datetime64')
        self.categorical = []
        for i in self.data.columns:
            if self.data[i].dtype =='object':
                self.categorical.append(i)
    def transform(self):
        self.O = OneHotEncoder()
        categorical_values = self.data.loc[:,self.categorical]
        categorical_values_ = self.O.fit_transform(categorical_values).toarray().astype(np.int64)
        # [, ] or < - change
        memo_columns = {i:'feature%d'%(_) for _,i in enumerate(self.O.get_feature_names())}
        columns = memo_columns.values()
        categorical_values_ = df(categorical_values_,columns=columns)
        numerical_values = self.data.loc[:,self.data.columns.difference(self.categorical)]
        
        pd.concat([categorical_values_,numerical_values],axis=1).to_csv(args.output_data,index=False)
        f = open('onehot_columns_memo','wb')
        pickle.dump(memo_columns,f)
        f.close()
        #train_cv.columns.difference(['Data60','Data61','Data62','Data63','Data57'])]
if __name__=='__main__':
    args=parser.parse_args()
    d = dataset()
    d.transform()  
