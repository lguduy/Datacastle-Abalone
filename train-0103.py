# -*- coding: utf-8 -*-
"""
Created on 2017/01/03 17:00

对梯度提升树算法进行调参
交叉验证，网格搜索

@author: lguduy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import explained_variance_score, make_scorer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", color_codes=True)


# 载入数据
filename = r'/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Data/traindata .csv'
data = pd.read_csv(filename)


# 将标称型数据转化为数值型数据（简单转换）
data.loc[data['Sex'] == 'M', 'Sex'] = 1
data.loc[data['Sex'] == 'F', 'Sex'] = 2
data.loc[data['Sex'] == 'I', 'Sex'] = 3


# 画箱线图，数据清洗
plt.figure(1, figsize=(9, 8))
box_plot = data.drop(['Rings'], axis=1).boxplot(return_type='dict')
plt.title('Boxplot')
plt.xticks(rotation=30)
fig_box = r'/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Fig/fig-box'
#plt.savefig(filename=fig_box, dpi=300)
plt.show()
plt.close()

data = data[data[u'Hight(mm)'] < 0.6]                    # 剔除异常数据点


"""Train and Test Data"""
X_data = data.drop(['Rings'], axis=1)
y_data = data['Rings']

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,
                                                    test_size=0.1,
                                                    random_state=0)

"""Cross validation and Grid search 选择最优参数"""
gbrt1 = GradientBoostingRegressor()                      # 梯度提升树模型

scorer = make_scorer(explained_variance_score)           # 模型评价标准


"""第一次网格搜索"""
param_grid1 = {'n_estimators': xrange(100, 501, 100),
               'learning_rate': np.arange(0.01, 0.11, 0.01)}


gbrt_gridsearch1 = GridSearchCV(estimator=gbrt1,
                                param_grid=param_grid1,
                                scoring=scorer,
                                n_jobs=-1,
                                cv=5,
                                verbose=1)

gbrt_gridsearch1.fit(X_train, y_train)                   # 训练

best_param1 = gbrt_gridsearch1.best_params_              # Best parameters
print "Best parameters is : {}".format(best_param1)

mean_test_score1 = gbrt_gridsearch1.cv_results_['mean_test_score']
mean_train_score1 = gbrt_gridsearch1.cv_results_['mean_train_score']

y_pre1 = gbrt_gridsearch1.predict(X_test)                # 默认用最好的模型预测

score1 = explained_variance_score(y_test, y_pre1)        # 预测结果评价
print 'Predict  score is : {}'.format(score1)


"""第二次网格搜索"""
gbrt2 = GradientBoostingRegressor(n_estimators=400,
                                  learning_rate=0.03)

param_grid2 = {'max_depth': xrange(2, 8),
               'min_samples_split': xrange(2, 102, 10)}

gbrt_gridsearch2 = GridSearchCV(estimator=gbrt2,
                                param_grid=param_grid2,
                                scoring=scorer,
                                n_jobs=-1,
                                cv=5,
                                verbose=1)

gbrt_gridsearch2.fit(X_train, y_train)                   # 训练

best_param2 = gbrt_gridsearch2.best_params_              # Best parameters
print "Best parameters is : {}".format(best_param2)

best_test_score2 = gbrt_gridsearch2.cv_results_['mean_test_score'].max()
best_train_score2 = gbrt_gridsearch2.cv_results_['mean_train_score'].max()

y_pre2 = gbrt_gridsearch2.predict(X_test)                # 默认用最好的模型预测

score2 = explained_variance_score(y_test, y_pre2)        # 预测结果评价
print 'Predict  score is : {}'.format(score2)


"""第三次网格搜索"""
gbrt3 = GradientBoostingRegressor(n_estimators=400,
                                  learning_rate=0.03,
                                  max_depth=3)

param_grid3 = {'min_samples_split': xrange(20, 101, 10),
               'min_samples_leaf': xrange(1, 20, 2)}

gbrt_gridsearch3 = GridSearchCV(estimator=gbrt3,
                                param_grid=param_grid3,
                                scoring=scorer,
                                n_jobs=-1,
                                cv=5,
                                verbose=1)

gbrt_gridsearch3.fit(X_train, y_train)                   # 训练

best_param3 = gbrt_gridsearch3.best_params_              # Best parameters
print "Best parameters is : {}".format(best_param3)

best_test_score3 = gbrt_gridsearch3.cv_results_['mean_test_score'].max()
best_train_score3 = gbrt_gridsearch3.cv_results_['mean_train_score'].max()

y_pre3 = gbrt_gridsearch3.predict(X_test)                # 默认用最好的模型预测

score3 = explained_variance_score(y_test, y_pre3)        # 预测结果评价
print 'Predict  score is : {}'.format(score3)


"""对max_features网格搜索"""
best_param_max_features = {'n_estimators': 400,
                           'learning_rate': 0.03,
                           'max_depth': 3,
                           'min_samples_split': 70,
                           'min_samples_leaf': 20}

gbrt4 = GradientBoostingRegressor(**best_param_max_features)

param_grid4 = {'max_features': xrange(1, 9, 1)}

gbrt_gridsearch4 = GridSearchCV(gbrt4,
                                param_grid=param_grid4,
                                scoring=scorer,
                                n_jobs=-1,
                                cv=5,
                                verbose=1)

gbrt_gridsearch4.fit(X_train, y_train)                   # 训练

best_param4 = gbrt_gridsearch4.best_params_              # Best parameters
print "Best parameters is : {}".format(best_param4)

best_test_score4 = gbrt_gridsearch4.cv_results_['mean_test_score'].max()
best_train_score4 = gbrt_gridsearch4.cv_results_['mean_train_score'].max()

y_pre4 = gbrt_gridsearch4.predict(X_test)                # 默认用最好的模型预测

score4 = explained_variance_score(y_test, y_pre4)        # 预测结果评价
print 'Predict  score is : {}'.format(score4)


"""对Subsamples网格搜索"""
gbrt5 = GradientBoostingRegressor(n_estimators=400,
                                  learning_rate=0.03,
                                  max_depth=3,
                                  min_samples_split=70,
                                  min_samples_leaf=20,
                                  max_features=4)

param_grid5 = {'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]}

gbrt_gridsearch5 = GridSearchCV(gbrt5,
                                param_grid=param_grid5,
                                scoring=scorer,
                                n_jobs=-1,
                                cv=5,
                                verbose=1)

gbrt_gridsearch5.fit(X_train, y_train)                   # 训练

best_param5 = gbrt_gridsearch5.best_params_              # Best parameters
print "Best parameters is : {}".format(best_param5)

best_test_score5 = gbrt_gridsearch5.cv_results_['mean_test_score'].max()
best_train_score5 = gbrt_gridsearch5.cv_results_['mean_train_score'].max()

y_pre5 = gbrt_gridsearch5.predict(X_test)                # 默认用最好的模型预测

score5 = explained_variance_score(y_test, y_pre5)        # 预测结果评价
print 'Predict  score is : {}'.format(score5)


"""交叉验证得到的最优参数"""
best_params =  {'n_estimators': 400,
                'learning_rate': 0.03,
                'max_depth': 3,
                'min_samples_split': 70,
                'min_samples_leaf': 20,
                'max_features': 4,
                'subsample': 0.6}

gbrt_best = GradientBoostingRegressor(**best_params)

"""用训练数据训练的模型"""
gbrt_best.fit(X_train, y_train)                           # 训练

y_pre_best = gbrt_best.predict(X_test)                    # 默认用最好的模型预测

score_best = explained_variance_score(y_test, y_pre_best) # 预测结果评价
print 'Predict  score is : {}'.format(score_best)

"""模型持久化"""
model_file_01 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-01"
joblib.dump(gbrt_best, model_file_01)


"""用全部已知数据训练的模型"""
gbrt_best_final = GradientBoostingRegressor(**best_params)

gbrt_best_final.fit(X_data, y_data)

"""模型持久化"""
model_file_02 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-02"
joblib.dump(gbrt_best_final, model_file_02)


"""用全部数据训练原始模型"""
gbrt_orig = GradientBoostingRegressor()

gbrt_orig.fit(X_data, y_data)

"""模型持久化"""
model_file_03 = r"/home/liangyu/Python/Data-Competitions/Datacastle/Project1/Model/model-GBDT-0106-orig"
joblib.dump(gbrt_best_final, model_file_03)
