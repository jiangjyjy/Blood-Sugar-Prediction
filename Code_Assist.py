# Code_Assist.py
# 辅助代码，有可能会报错，但是可以运行出结果

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
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import variance_threshold
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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
d_train = pd.read_csv(train_datafile,encoding='gbk')
test_dataA = pd.read_csv(test_datafileA,encoding='gbk')
test_dataB = pd.read_csv(test_datafileB,encoding='gbk')
answer_dataA = pd.read_csv(answer_datafileA,encoding='gbk')
answer_dataB = pd.read_csv(answer_datafileB,encoding='gbk')
d_test=pd.concat([test_dataA,test_dataB],axis=0,sort=False)
answer_data=pd.concat([answer_dataA,answer_dataB],axis=0,sort=False)


# 数据预处理
# 数据准备
d_train = d_train.drop('id', axis=1)  # 将id列删除
d_train['性别'] = d_train['性别'].map({'男': 1, '女': 0, '??': 1})  # 将男女进行二值转化
d_train.groupby(['性别']).size().plot.pie(subplots=True, autopct='%.2f', fontsize=20)

d_test = d_test.drop('id', axis=1)
d_test['性别'] = d_test['性别'].map({'男': 1, '女': 0, '??': 1})
d_test.groupby(['性别']).size().plot.pie(subplots=True, autopct='%.2f', fontsize=20)


"""
缺失值处理
"""
# 将字段名更改为 var_0-40（避免乱码问题）
for i in range(d_train.shape[1]):
    d_train = d_train.rename(columns={d_train.columns[i]: 'var_' + str(i)})
for i in range(d_test.shape[1]):
    d_test = d_test.rename(columns={d_test.columns[i]: 'var_' + str(i)})

# 画图查看缺失值分布情况
nulls = d_train.isnull().sum().sort_values(ascending=False)
nulls.plot(kind='bar')
(nulls / d_train.shape[0]) * 100

# 删除缺失值占比超过70%的字段(乙肝检查)
remove1 = ["var_18", "var_19", "var_20", "var_21", "var_22"]
train_1 = d_train.drop(remove1, axis=1)
test_1 = d_test.drop(remove1, axis=1)
train_1.shape

(train_1.isnull().sum().sort_values(ascending=False) / train_1.shape[0]) * 100

# 查看极值
des = train_1.quantile([0, 0.01, 0.1, 0.5, 0.8, 0.9, 0.99, 1])
des

# 画图查看极值
for i in range(des.shape[1]):
    # i = 1
    plt.plot(des.iloc[:, i], c='b')
    plt.ylabel(des.columns[i])
    plt.show()

# 查看一下血糖分布
plt.plot(train_1.loc[:, 'var_40'])
plt.hist(train_1.loc[:, 'var_40'], alpha=0.5, bins=50)

# 用随机森林对其他缺失值进行填补
train_1 = train_1.drop('var_2', axis=1)
x_missing = train_1.copy()
x_missing = x_missing.drop('var_0', axis=1)
missing = x_missing.isnull().sum().sort_values(ascending=False)

for i in missing.index:
    # i = 'var_16'
    y = x_missing[str(i)]
    x = x_missing.drop(str(i), axis=1)
    if len(y) == 0:
        break
    # 将其他缺失值用0填充
    x_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(x)
    ytrain = y[y.notnull()]
    ytest = y[y.isnull()]

    xtrain = x_0[ytrain.index, :]
    xtest = x_0[ytest.index, :]

    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(xtrain, ytrain)
    ypredict = rfc.predict(xtest)

    # 将填好的特征返回到原始特征矩阵中
    x_missing.loc[x_missing.loc[:, str(i)].isnull(), str(i)] = ypredict

train_2 = x_missing.copy()
train_2.isnull().sum()