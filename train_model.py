# -*- coding: utf-8 -*-
import pandas as pd
from pandas import DataFrame as df
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score#,mean_absolute_error
import math
import os
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import pickle

parser = argparse.ArgumentParser(description = 'model train')

parser.add_argument('--test_size', type = float , default = 0.3 , help = 'test_size')
parser.add_argument('--test_type', type = str , default = 'sequential' , help = 'sequential이면 마지막 test size만큼을 남김')
parser.add_argument('--data', type = str , default = './data/preprocessed_data.csv' , help = '데이터 저장 위치')
parser.add_argument('--report', type = str , default = 'report' , help = '데이터 저장 위치')
parser.add_argument('--target_feature_names',type = list, default = ['Data60','Data61','Data62','Data63'], help ='종속변수들')
parser.add_argument('--except_feature_names',type = list, default = ['Data57'], help ='제외시킬 변수들')


def MAPE(y_true,y_pred):
    '''
    Returns
    -------
    mean absolute percentage error
    formula = mean(abs((y_true - y_pred)/(y_true)))*100
    y_true, y_pred : (N,)
    '''
    return np.mean(abs((y_true-y_pred)/(y_true+1e-8)))*100

class train_model:
    def __init__(self):
        self.scores = {}
        for target in args.target_feature_names:
            self.scores[target]={}
        self.save_model = {target:None for target in args.target_feature_names}
    def train(self):
        for target in args.target_feature_names:
            model = LGBMRegressor() # 여기만 변경해주면 됨.
            
            self.scores[target]={'train':{},'test':{}}
            train_y = train.loc[:,target] 
            train_x = train.loc[:,train.columns.difference(args.target_feature_names+args.except_feature_names)]
            test_y = test.loc[:,target] 
            test_x = test.loc[:,test.columns.difference(args.target_feature_names+args.except_feature_names)]
            model.fit(train_x,train_y)
            train_pred = model.predict(train_x)
            self.scores[target]['train']['rmse']=np.sqrt(mean_squared_error(train_y,train_pred))
            self.scores[target]['train']['mape']=MAPE(train_y,train_pred)
            self.scores[target]['train']['r2']=r2_score(train_y,train_pred)
            test_pred = model.predict(test_x)
            self.scores[target]['test']['rmse']=np.sqrt(mean_squared_error(test_y,test_pred))
            self.scores[target]['test']['mape']=MAPE(test_y,test_pred)
            self.scores[target]['test']['r2']=r2_score(test_y,test_pred)
            self.save_model[target]=model
            
            
    def save_scores(self):
        try:
            os.mkdir(args.test_type)
        except:
            pass
        os.chdir(args.test_type)
        
        
        for target in args.target_feature_names:
            d = {}
            for i in self.scores[target].keys(): # train,test
                y = []
                d[i]=[]
                for r,value in self.scores[target][i].items(): # mse,r2,mape
                    d[i].append(value)
                    y.append(r)
            report = df(d,index = y) 
            report.to_csv('./report+%s.csv'%target, index = True)
        
        
        os.chdir('..')
  
# train, test split

if __name__=='__main__':
    args=parser.parse_args()
    data = pd.read_csv(args.data,header=0,index_col=0)
    data['Data57']=data['Data57'].astype('datetime64')
    if args.test_type == 'sequential':
        shuffle = False
    else: 
        shuffle = True
    train,test=train_test_split(data,test_size=args.test_size,shuffle=shuffle)
    
    t = train_model()
    t.train()
    t.save_scores()
    F = open('./save_model','wb')
    pickle.dump(t,F)
    F.close()
    
    
    
