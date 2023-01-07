import pandas as pd
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoCV
from sklearn.model_selection import train_test_split


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

ridge_model=Ridge(alpha=0.188,random_state=2)
ridge_model.fit(X_train,y_train)
predict2=ridge_model.predict(X_test)
r2rid=mt.r2_score(y_test,predict2)
mserid=mt.mean_squared_error(y_test,predict2)

print("************Ridge Regression(alpha=0.188)************\nR2 Ridge: {}    \nMSE Ridge:{}\n************Ridge Regression(alpha=0.188)************\n".format(
    r2rid,mserid))

lasso_model=Lasso(random_state=2)                       
lasso_model.fit(X_train,y_train)                    
predictlasso=lasso_model.predict(X_test)

print("************Lasso Regression(Default)************\nR2 Lasso: {}    \nMSE Lasso:{}\n************Lasso Regression(Default)************\n".format(
    lasso_model.score(X_test,y_test),mt.mean_squared_error(y_test,predictlasso)))
################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Models and Default Scores Determined ###################


################### Optimized Alpha Detected and Applied to Lasso Model ###################
lambdas2=LassoCV(cv=10,  max_iter=10000).fit(X_train,y_train).alpha_   
lasso_model2=Lasso(alpha=lambdas2,random_state=2)                                     
lasso_model2.fit(X_train,y_train)                                      
predict2lasso=lasso_model2.predict(X_test)
print("************Lasso Regression(Optimized Alpha)************\nR2 Lasso:{}    \nMSE Lasso:{}\n************Lasso Regression(Optimized Alpha)************\n".format(
    lasso_model2.score(X_test,y_test), mt.mean_squared_error(y_test,predict2lasso)))
################### Optimized Alpha Detected and Applied to Ridge Model ###################


################### Model Coefficients Printed for All Models ###################
print("********************************Coefficents********************************\n",
"Linear Regression:{}\n Ridge Regression:{}\n Lasso Regression(Optimized Alpha):{}".format(Lr.coef_,ridge_model.coef_,lasso_model2.coef_),
"\n********************************Coefficents********************************")
################### Model Coefficients Printed for All Models ###################