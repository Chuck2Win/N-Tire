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
parser.add_argument('--models', type = list , default = ['rf','xg','lgbm'] , help = '모델')
parser.add_argument('--test_size', type = float , default = 0.3 , help = 'test_size')
parser.add_argument('--test_type', type = str , default = 'sequential' , help = 'sequential이면 마지막 test size만큼을 남김')
#parser.add_argument('--k', type = int , default = 10 , help = 'cross validation')
parser.add_argument('--data', type = str , default = 'G:/넥센타이어/Foot Print/preprocessed_data.csv' , help = '데이터 저장 위치')
parser.add_argument('--report', type = str , default = 'report' , help = '데이터 저장 위치')
parser.add_argument('--memo', type = str , default = 'G:/넥센타이어/Foot Print/onehot_columns_memo' , help = 'Onehotvector 이름')
parser.add_argument('--ys',type = list, default = ['Data60','Data61','Data62','Data63'], help ='종속변수들')
parser.add_argument('--except_x',type = list, default = ['Data57'], help ='불필요한 변수들')


def MAPE(y_true,y_pred):
    '''
    Returns
    -------
    mean absolute percentage error
    formula = mean(abs((y_true - y_pred)/(y_true)))*100
    y_true, y_pred : (N,)
    '''
    return np.mean(abs((y_true-y_pred)/(y_true+1e-8)))*100

class train_models:
    def __init__(self):
        self.scores = {}
        self.models = args.models
        for target in args.ys:
            self.scores[target]={}
        self.save_models = {name:None for name in self.models}
    def train(self):
        for target in args.ys:
            for name in self.models:
             # 현행 방향 - 종속변수끼리의 교호 작용 파악 x
                self.scores[target][name]={'train':{},'test':{}}
                train_y = train.loc[:,target] 
                train_x = train.loc[:,train.columns.difference(args.ys+args.except_x)]
                test_y = test.loc[:,target] 
                test_x = test.loc[:,test.columns.difference(args.ys+args.except_x)]
                if name == 'xg':
                    model = XGBRegressor()
                elif name =='rf':
                    model = RandomForestRegressor(max_features=int(math.sqrt(train_x.shape[1]/3)), min_samples_leaf=1,max_depth=30,n_estimators=100, oob_score=True, n_jobs=-1)
                elif name == 'lgbm':
                    model = LGBMRegressor()
                model.fit(train_x,train_y)
                train_pred = model.predict(train_x)
                self.scores[target][name]['train']['mse']=mean_squared_error(train_y,train_pred)
                self.scores[target][name]['train']['mape']=MAPE(train_y,train_pred)
                self.scores[target][name]['train']['r2']=r2_score(train_y,train_pred)
                test_pred = model.predict(test_x)
                self.scores[target][name]['test']['mse']=mean_squared_error(test_y,test_pred)
                self.scores[target][name]['test']['mape']=MAPE(test_y,test_pred)
                self.scores[target][name]['test']['r2']=r2_score(test_y,test_pred)
                self.save_models[name]=model
    def save_scores(self):
        try:
            os.mkdir(args.test_type)
        except:
            pass
        os.chdir(args.test_type)
        for target in args.ys:
            d = {}
            for i in self.scores[target].keys(): # model
                y = []
                d[i]=[]
                for j in self.scores[target][i].keys(): # train, test
                    
                    for r,value in self.scores[target][i][j].items(): # mse,r2,mape
                        d[i].append(value)
                        y.append(j+'_'+r)
            report = df(d,index = y) 
            report.to_csv('./report+%s.csv'%target, index = True)
        os.chdir('..')
  
# train, test split
# Kfold Validation
if __name__=='__main__':
    args=parser.parse_args()
    data = pd.read_csv(args.data,header=0)
    data['Data57']=data['Data57'].astype('datetime64')
    if args.test_type == 'sequential':
        train = data.loc[data['Data57']<=data['Data57'][int(len(data)*args.test_size)],:]
        test = data.loc[data['Data57']>data['Data57'][int(len(data)*args.test_size)],:]
    else: 
        train,test=train_test_split(data,test_size = args.test_size, random_state = 123, shuffle = True)
    t = train_models()
    t.train()
    t.save_scores()
    t.scores
    os.chdir(args.test_type)
    f = open('train_model_%s'%args.test_type,'wb')
    pickle.dump(t,f)
    f.close()
    os.chdir('..')
