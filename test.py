# test.py

# 引入必要的模块
import xgboost as xgb
import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import time

# 数据准备与必要引用的机器学习模块
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.writer.excel import ExcelWriter


# 解决matplotlib显示中文问题
# 仅适用于Windows
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题



# 指定数据集路径
dataset_path = 'C:/Users/yue/Desktop/Blood sugar to predict'
train_datafile = os.path.join(dataset_path, 'd_train_20180102.csv')
test_datafileA = os.path.join(dataset_path, 'd_test_A_20180102.csv')
test_datafileB= os.path.join(dataset_path, 'd_test_B_20180128.csv')
answer_datafileA= os.path.join(dataset_path, 'd_answer_a_20180128.csv')
answer_datafileB= os.path.join(dataset_path, 'd_answer_b_20180130.csv')
# 加载数据
train_data = pd.read_csv(train_datafile,encoding='gbk')
test_dataA = pd.read_csv(test_datafileA,encoding='gbk')
test_dataB = pd.read_csv(test_datafileB,encoding='gbk')
answer_dataA = pd.read_csv(answer_datafileA,encoding='gbk')
answer_dataB = pd.read_csv(answer_datafileB,encoding='gbk')
test_data=pd.concat([test_dataA,test_dataB],axis=0,sort=False)#将两个测试集合并
answer_data=pd.concat([answer_dataA,answer_dataB],axis=0,sort=False)#将两个测试集答案合并


#数据预处理
# 数据准备
train_data=train_data.drop('id', axis=1)
train_data=train_data.drop('体检日期', axis=1)
train_data['性别']=train_data['性别'].map({'男': 1,'女': 0, '??':1})
test_data=test_data.drop('id', axis=1)
test_data=test_data.drop('体检日期', axis=1)
test_data['性别']=test_data['性别'].map({'男': 1,'女': 0, '??':1})

# 去除缺失比例大于百分之70的值
remove=["乙肝e抗原","乙肝核心抗体","乙肝表面抗原","乙肝表面抗体","乙肝e抗体"]
train_data=train_data.drop(remove, axis=1)
test_data = test_data.drop(remove, axis=1)

# 移除异常值
# 训练集
columns=len(train_data.columns)
train_data.drop(train_data.index[[i for i in train_data.index if train_data.iloc[i,columns-1]>30]],inplace=True)
# 测试集
columns=len(test_data.columns)
test_data.drop(test_data.index[[i for i in test_data.index if test_data.iloc[i,columns-1]>30]],inplace=True)

# 缺失值填充为均值
train_data.fillna(train_data.mean(),inplace=True)
test_data.fillna(test_data.mean(),inplace=True)



# 特征向量
# 训练集
feat_names = train_data.columns[:-1].tolist()  # 去除血糖值
X_train = train_data[feat_names].values
feat_names=train_data.columns[-1:].tolist()  # 血糖值
y_train = train_data[feat_names].values
# 测试集
feat_names = test_data.columns[:].tolist()
X_test = test_data[feat_names].values
feat_names = answer_data.columns[:].tolist()
y_test = answer_data[feat_names].values





# 使用sklearn中xgboost进行建模
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
params = {'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'max_depth': 6,  # 通常取值：3-10
            'gamma':0.2,  # 给定了所需的最低loss function的值
            'lambda': 100,
            'subsample': 1,  # 用于训练模型的子样本占整个样本集合的比例
            'colsample_bytree': 0.6,
            'min_child_weight': 12,  # 5~10,孩子节点中最小的样本权重和，即调大这个参数能够控制过拟合
            'eta': 0.02,  # 更新过程中用到的收缩步长，取值范围为：[0,1]
            'sample_type': 'uniform',
            'normalize': 'tree',
            'rate_drop': 0.1,
            'skip_drop': 0.9,
            'seed': 100,
            'nthread':-1
            }
bst_nb = 700
watchlist = [(dtrain, '训练误差')]
model = xgb.train(params, dtrain, num_boost_round=bst_nb, evals=watchlist)  # 训练模型
y_pred = model.predict(dtest)
print(y_pred)

# 模型评估
summ=0
for i in range(len(y_pred)):
     summ = summ + (y_pred[i]-y_test[i])**2
summ_erro = round(float(summ/(2*len(y_pred))), 4)
print("均方误差:", summ_erro)