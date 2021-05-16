#
# Created by Zs on 21-5-1
#

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from zspytorch import boston


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    score = r2_score(y_true, y_predict)

    # Return the score
    return score


data = load_boston()
# print(type(data))
data_pd = pd.DataFrame(data.data, columns=data.feature_names)
data_pd['price'] = data.target
# 相关系数
Corr = data_pd.corr()
Corr_price = Corr['price']

# 划分数据集和测试集
alldata = data_pd.copy()
data_pd = data_pd[['LSTAT', 'PTRATIO', 'RM', 'price']]
y = np.array(data_pd['price'])
# data_pd = data_pd.drop(['price'], axis=1)
x = np.array(data_pd[['LSTAT', 'PTRATIO', 'RM']])
y_all = np.array(alldata['price'])
x_all = np.array(alldata.drop(['price'], axis=1))

r2_score_list = []
temp_list = []

Regression_dict = {'Linreg': LinearRegression(),
                   'Ridge': Ridge(),
                   'Lasso': Lasso(),
                   'ElasticNet': ElasticNet(),
                   'DecisionTree': DecisionTreeRegressor(),
                   'RandomForest': RandomForestRegressor(),
                   'ExtraTrees': ExtraTreesRegressor(),
                   'GradientBoost': GradientBoostingRegressor()
                   }
Regression_list = ['Linreg', 'Ridge', 'Lasso', 'ElasticNet', 'DecisionTree', 'RandomForest', 'ExtraTrees',
                   'GradientBoost']
"""
for Regression in Regression_list:
    model = Regression_dict[Regression]
    temp_list = []
    for i in range(20):
        train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)
        model.fit(train_X, train_Y)
        y_predict = model.predict(test_X)
        temp_list.append(performance_metric(y_predict, test_Y))
    r2_score_list.append(temp_list.copy())
"""

for Regression in Regression_list:
    model = Regression_dict[Regression]
    temp_list = []
    for i in range(20):
        train_X, test_X, train_Y, test_Y = train_test_split(x_all, y_all, test_size=0.2)
        model.fit(train_X, train_Y)
        y_predict = model.predict(test_X)
        temp_list.append(performance_metric(y_predict, test_Y))
    r2_score_list.append(temp_list.copy())

load_weights = True
load_cols = []
input_shape = 13
if len(load_cols) != 0:
    input_shape = len(load_cols)

bos = boston()
x, y = bos.load_data(choose_col=load_cols)
bos.load_model(weights_name='Boston.pt', learn_rate=0.01)
temp_list = []
for i in range(20):
    train_x, train_y, test_x, test_y = bos.split_data(x=x, y=y, split_size=0.2)

    y_predict = bos.predict(test_x)
    temp_list.append(performance_metric(y_predict, test_y))
r2_score_list.append(temp_list)

# y_predict = linreg.predict(test_X)
# grady_predict = gradientreg.predict(test_X)

Regression_list.append('MLP')
plt.boxplot(r2_score_list)
scale = range(1, 10)
plt.xticks(scale, Regression_list, rotation=10)
# plt.tick_params(labelsize=8)
plt.ylabel('r2_score')
plt.xlabel('Regression_model')
plt.title('Regression model comparison(6 independent variable)')
plt.tight_layout()
plt.savefig('model_compar(addmlp).jpg', dpi=400)
plt.show()
