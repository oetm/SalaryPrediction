import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.metrics import *
import numpy as np

def line(x, a, b):
    y = 0
    for i in range(0, len(a)):
        y = y + np.asarray(x[i]) * a[i]

    return y + b

df = pd.read_csv('kc_house_data.csv', delimiter=',', usecols=['price',
                                                              'sqft_living',
                                                              'bathrooms',
                                                              'grade'])

correlation_matrix = df.corr()
sb.heatmap(correlation_matrix, annot=True)

x = df[['sqft_living', 'bathrooms', 'grade']].values
#x = df[['grade']].values
y = df['price'].values

y_mean = y.mean()
y_std = y.std()

y = (y - y_mean)/y_std

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

mod = LinearRegression()
reg = mod.fit(x_train, y_train)

y_pred = reg.predict(x_test)
y_train_pred = reg.predict(x_train)

print("R^2 for train data {}".format(reg.score(x_train, y_train)))
print("Mean Squared Error for train data {}".format(mean_squared_error(y_train, y_train_pred)))
print("R^2 for test data {}".format(reg.score(x_test, y_test)))
print("Mean Squared Error for test data {}".format(mean_squared_error(y_test, y_pred)))

a = reg.coef_
b = reg.intercept_
X = []

for i in range(0, len(x[0, :])):
    rg = [x[:, i].min(), x[:, i].max()]
    X.append(rg)

curve = line(X, a, b)