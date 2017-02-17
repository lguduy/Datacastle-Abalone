# -*- coding: utf-8 -*-
"""
Created on 2017/01/04 19:34

载入训练好的GBDT模型作预测

@author: lguduy
"""

from sklearn.externals import joblib
import pandas as pd
import numpy as np


"""载入训练好的模型"""
model_file_01 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-01"
gbrt_best_traindata = joblib.load(model_file_01)

model_file_02 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-02"
gbrt_best_alldata = joblib.load(model_file_02)

model_file_03 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-orig"
gbrt_best_orig = joblib.load(model_file_03)

xgb_model_01 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/xgb_model-0116-01"
xgb_model = joblib.load(xgb_model_01)


"""载入测试数据"""
filename = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Data/testdata.csv"
X = pd.read_csv(filename, index_col="ID")

X.loc[X['Sex'] == 'M', 'Sex'] = 1
X.loc[X['Sex'] == 'F', 'Sex'] = 2
X.loc[X['Sex'] == 'I', 'Sex'] = 3
X["Sex"] = X["Sex"].astype(np.int64)      # 改变列的类型
X.columns = [u'Sex', u'Length(mm)', u'Diameter(mm)', u'Hight(mm)',
       u'Whole weight(g)', u'Shucked weight(g)', u'Viscera weight(g)',
       u'Shell weight(g)']


"""预测"""
y_pred = xgb_model.predict(X)

m = y_pred.shape[0]

y_pred_df = pd.DataFrame(y_pred, index=xrange(1, m+1))
y_pred_df_int = y_pred_df.apply(lambda x: round(x), axis=1)
y_pred_df_int_df = pd.DataFrame(y_pred_df_int)

predict_filename = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Data/predict_round0116-01.csv"
y_pred_df_int_df.to_csv(predict_filename, index_label="ID", header=["Rings"])
