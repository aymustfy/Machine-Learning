import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,RidgeCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt


################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Models and Default Scores Determined ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Ecommerce Customers.csv")
data2=data.copy()

y=data2["Yearly Amount Spent"]
X=data2[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Lr=LinearRegression()
Lr.fit(X_train,y_train)
predictions=Lr.predict(X_test)

r2=mt.r2_score(y_test,predictions)
mse=mt.mean_squared_error(y_test,predictions)
rmse=mt.mean_squared_error(y_test,predictions,squared=False)
mae=mt.mean_absolute_error(y_test,predictions)

print("\n************Linear Regression************\nR2:{}\nMSE:{}\nRMSE:{}\nMAE:{}\n************Linear Regression************\n".format(r2,mse,rmse,mae))

ridge_model=Ridge()
ridge_model.fit(X_train,y_train)
predict2=ridge_model.predict(X_test)
r2rid=mt.r2_score(y_test,predict2)
mserid=mt.mean_squared_error(y_test,predict2)
print("************Ridge Regression(Default)************\nR2rid: {}    \nMSErid:{}\n************Ridge Regression(Default)************\n".format(
    r2rid,mserid))
################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Models and Default Scores Determined ###################


################### Optimized Alpha Detected and Applied to Ridge Model ###################
lambdas=10**np.linspace(10,-2,100)*0.5
ridge_cv=RidgeCV(alphas=lambdas,scoring="r2")
ridge_cv.fit(X_train,y_train)
print("\nOptimized Alpha>>>>>>>>>",ridge_cv.alpha_)

ridge_model2=Ridge(alpha=ridge_cv.alpha_)
ridge_model2.fit(X_train,y_train)
predict3=ridge_model2.predict(X_test)
r2rid2=mt.r2_score(y_test,predict3)
mserid2=mt.mean_squared_error(y_test,predict3)
print("\n\n************Ridge Regression(Optimized Alpha)************\nR2rid: {}    \nMSErid:{}\n************Ridge Regression(Optimized Alpha)************".format(
    r2rid2,mserid2))
################### Optimized Alpha Detected and Applied to Ridge Model ###################


################### Model Scores and Coefficients Printed for All Models ###################
print("\n\n********************************Coefficents********************************\n",
"Linear Regression:{}\n Ridge Regression(Default):{}\n Ridge Regression(Optimized Alpha:{}".format(Lr.coef_,ridge_model.coef_,ridge_model2.coef_),
"\n********************************Coefficents********************************")

print("\n While the success rate of the Ridge Regression model is higher than other models, the rmse value is also low. ",
"Therefore, if the variable coefficients of that model are taken into account;",
"\n Avg. Session Length---->{}\n Time on App---->{}\n Time on Website---->{}\n Length of Membership---->{}".format(
    ridge_model.coef_[0],ridge_model.coef_[1],
ridge_model.coef_[2],ridge_model.coef_[3]))
################### Model Scores and Coefficients Printed for All Models ###################