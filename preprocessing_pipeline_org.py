# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 06:49:27 2021

@author: IDSL
"""
# pipeline

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# get params, set params // fit, transform, fit trasform
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline#FeatureUnion,

#  Feature Union은 np,array만 취급하는 듯.
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='pipeline')
#parser.add_argument('--data', type = str , default = ./footprint_v2.csv' , help = '데이터 저장 위치')
parser.add_argument('--preprocessed_data',type=str,default='./data/preprocessed_data_org.csv')
parser.add_argument('--outlier_percentile',type=float,default=0.01)
parser.add_argument('--dictionary',type=str,default='./data/dictionary')
parser.add_argument('--categorical_feature_names', type = list , default = 
['Data2',
 'Data3',
 'Data4',
 'Data5',
 'Data9',
 'Data25',
 'Data43',
 'Data44',
 'Data45',
 'Data47',
 'Data49',
 'Data52',
 'Data53',
 ])


parser.add_argument('--target_feature_names', type = list , default = ['Data60','Data61','Data62','Data63'])
parser.add_argument('--time_feature_name', type = list , default = ['Data57'])
parser.add_argument('--numerical_feature_names',type = list, default = [
 'Data6',
 'Data10',
 'Data11',
 'Data12',
 'Data13',
 'Data14',
 'Data15',
 'Data16',
 'Data17',
 'Data18',
 'Data19',
 'Data20',
 'Data21',
 'Data22',
 'Data23',
 'Data24',
 'Data26',
 'Data27',
 'Data28',
 'Data29',
 'Data30',
 'Data31',
 'Data32',
 'Data33',
 'Data34',
 'Data35',
 'Data36',
 'Data37',
 'Data38',
 'Data39',
 'Data40',
 'Data41',
 'Data42',
 'Data51',
 'Data58',
 'Data59'])

# 이상치 제거 - numerical variable을 위해


# target variable을 위해서
class DropRow(BaseEstimator, TransformerMixin):
    # y값이 결측이 있으면 버린다.
    # -값이라는 결측값도 있음.
    def __init__( self,feature_names):
        self._feature_names = feature_names 
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # 결측치 제거
        X=X.loc[X[self._feature_names].isna().sum(axis=1)==0,:]
        # -값 제거
        X=X.loc[((X[self._feature_names]=='-').sum(axis=1))==0,:]
       
        return X

# numerical 변수 + target variable을 위한 것임
# impute 된 후에 적용하면 됨
# normal 하게 2.5%~97.5%만을 get
class OulierTransformer(BaseEstimator, TransformerMixin):
    def __init__( self, outlier_percentile):
        self.outlier_percentile = outlier_percentile
        
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        # 결측치 제거
        idx = None
        for i in X.columns:
            a,b = np.percentile(X[i].values,[self.outlier_percentile, 100.0-self.outlier_percentile])
            if idx is None:
                idx = (a<=X[i]) & (X[i]<=b)
            else:
                idx = idx &( (a<=X[i]) & (X[i]<=b))
            
        return X.loc[idx,:]

        
class FeatureSelector( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__( self, feature_names ):
        self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None ):
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        return X[ self._feature_names ] 

# 시간으로 변환

class TimeTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        return self
        
    
    def transform(self,X,y=None):
        return X.astype('datetime64')

# combine  - FeatureUnion(pipeline을 합치기)

class CategoricalTransformer1(BaseEstimator, TransformerMixin):
    import pandas as pd
    def __init__(self):
        super().__init__()
        self.SimpleImputer=SimpleImputer(strategy = 'most_frequent')
        
    def fit(self,X,y=None):
        self.SimpleImputer.fit(X)
        return self # 자기 자신을 return 해야함.
    
    def transform(self,X,y=None):
        output = self.SimpleImputer.transform(X)
        df = pd.DataFrame(output,columns=X.columns,index = X.index)
        return df

class CategoricalTransformer2(BaseEstimator, TransformerMixin):
    import pandas as pd
    def __init__(self):
        super().__init__()
        self.OneHotEncoder=OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    def fit(self,X,y=None):
        self.OneHotEncoder.fit(X)
        return self # 자기 자신을 return 해야함.
    
    def transform(self,X,y=None):
        output = self.OneHotEncoder.transform(X)
        df = pd.DataFrame(output,columns=self.OneHotEncoder.get_feature_names(),index=X.index)
        return df
    
class NumericalTransformer(BaseEstimator, TransformerMixin):
    import pandas as pd
    def __init__(self):
        super().__init__()
        self.SimpleImputer=SimpleImputer(strategy = 'median')
        
    def fit(self,X,y=None):
        self.SimpleImputer.fit(X)
        return self # 자기 자신을 return 해야함.
    
    def transform(self,X,y=None):
        output = self.SimpleImputer.transform(X)
        df = pd.DataFrame(output,columns=X.columns,index=X.index)
         # inf값 제거
        df=df.loc[((df=='inf').sum(axis=1))==0,:]
        df=df.loc[((df==float('inf')).sum(axis=1))==0,:]
        
        return df
 
class Cleanse(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
        
    def fit(self,X,y=None):
        return self
    
    def cleanse(self,i):
        try:
            return float(i)
        except:
            result = i.split('|')[0]
            try: 
                return float(result)
            except: # 개중에 ..도 있음.
                result=result.split('.')
                return  float(result[0]+'.'+result[-1])
    def transform(self,X,y=None):
        for c in X.columns:
            X.loc[:,c] = X.loc[:,c].apply(lambda i : self.cleanse(i))
        X=X.loc[((X=='inf').sum(axis=1))==0,:]
        X=X.loc[((X==float('inf')).sum(axis=1))==0,:]
        return X

class MergePreprocessedData(BaseEstimator, TransformerMixin):
    import pandas as pd
    def __init__(self,args):
        super().__init__()
        
        self.categorical_pipeline = Pipeline(steps = [('select', FeatureSelector(args.categorical_feature_names)),
                                             ('imputer', CategoricalTransformer1()),
                                             ('transform', CategoricalTransformer2())])
        self.numerical_pipeline = Pipeline(steps = [('select', FeatureSelector(args.numerical_feature_names)),
                                            ('transform',NumericalTransformer()),('outlier',OulierTransformer(args.outlier_percentile))])
        self.time_pipeline = Pipeline(steps=[('select',FeatureSelector(args.time_feature_name)),
                                    ('transform',TimeTransformer())])
        self.target_pipeline = Pipeline(steps=[('select',FeatureSelector(args.target_feature_names)),
                                      ('cleanse',Cleanse()),('outlier',OulierTransformer(args.outlier_percentile))])
        
    def fit(self,X):
        
        self.categorical_pipeline.fit(X)
        self.numerical_pipeline.fit(X)
        self.time_pipeline.fit(X)
        self.target_pipeline.fit(X)
        return self
        
    def transform(self,X):
        import pickle
        c=self.categorical_pipeline.transform(X)
        # category column은 수정해줘야함
        dictionary = {i:'feature_%d'%_  for _,i in enumerate(c.columns)}
        c.columns = [dictionary[i] for i in c.columns]
        F = open(args.dictionary,'wb')
        pickle.dump(dictionary,F)
        F.close()
        n=self.numerical_pipeline.transform(X)
        tm=self.time_pipeline.transform(X)
        t=self.target_pipeline.transform(X)
        
        idx1 = n.index
        idx2 = t.index
        idx = idx1&idx2
        c = c.loc[idx,:]
        n = n.loc[idx,:]
        tm = tm.loc[idx,:]
        t = t.loc[idx,:]
        
        standard = c
        for i in [n,tm,t]:
            standard = standard.join(i)
        return standard
    
    
    

if __name__=='__main__':
    raw_data = pd.read_excel('./Foot Print/FootPrintData_v1.0.xlsx',header=0)
    args = parser.parse_args()
    
    target_pipeline = Pipeline(steps=[('select',FeatureSelector(args.target_feature_names)),
                                      ('cleanse',Cleanse()),('outlier',OulierTransformer(args.outlier_percentile))])
    drop_pipeline = DropRow(args.target_feature_names)    
    final_pipeline = Pipeline(steps=[('drop_pipeline',drop_pipeline),('full_pipeline',MergePreprocessedData(args))])
    final_data = final_pipeline.fit_transform(raw_data)
    # 시간 순으로 sort
    final_data=final_data.sort_values(by=args.time_feature_name[0],axis=0) 
    final_data.to_csv(args.preprocessed_data,index=True)

   