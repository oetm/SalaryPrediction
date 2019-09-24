import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.linear_model import *

def line(x,a,b):
    return a*x + b

df = pd.read_csv('Salary.csv', delimiter = ',') #reading the data

x = df['YearsExperience'].values
x = x.reshape(-1, 1)
y = df['Salary'].values
y = y.reshape(-1, 1)

y_mean = y.mean()
y_std = y.std()

y = (y - y_mean) / y_std #data normalizing

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2) #splitting the data into train and test data

plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test)

plt.xlabel('Years of Experience') #in order to decide if the linear regression model is suitable for this situation, weneed to plot the data first
plt.ylabel('Salary')

correlation_matrix = df.corr()
print(correlation_matrix)

lr = LinearRegression()
reg = lr.fit(x_train, y_train) #perform the training step

y_pred = reg.predict(x_test) #perform the prediction step
y_train_pred = reg.predict(x_train)

print("R^2 for train data {}".format(reg.score(x_train, y_train)))
print("Mean Squared Error for train data {}".format(mean_squared_error(y_train, y_train_pred)))
print("R^2 for test data {}".format(reg.score(x_test, y_test)))
print("Mean Squared Error for test data {}".format(mean_squared_error(y_test, y_pred)))

a = reg.coef_[0]
b = reg.intercept_

X = [x.min(), x.max()]

plt.plot(X, line(X,a,b), c = 'red')
plt.show()