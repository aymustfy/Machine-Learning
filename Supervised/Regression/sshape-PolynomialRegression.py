import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import sklearn.metrics as mt



data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Polynomial Regression/sshape.csv")

plt.scatter(data["year"],data["price"],c="Blue")
plt.xlabel("year")
plt.ylabel("price")
plt.show()

y=data["price"]
X=data["year"]

y=y.values.reshape(-1,1)
X=X.values.reshape(-1,1)


pol=PolynomialFeatures(degree=3)
X_pol=pol.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(X_pol,y, test_size=0.2,
random_state=42)
pol_reg=LinearRegression()
pol_reg.fit(X_train,y_train)
predict=pol_reg.predict(X_test)
r21=mt.r2_score(y_test,predict)
mse1=mt.mean_squared_error(y_test,predict)
print("Polynomial R2={}      Polynomial MSE={}".format(r21,mse1))


lr2=LinearRegression()
lr2.fit(X_pol,y)
predict2=lr2.predict(X_pol)
r2pol=mt.r2_score(y,predict2)
msepol=mt.mean_squared_error(y,predict2)
print("Polynomial R2={}      Polynomial MSE={}".format(r2pol,msepol))