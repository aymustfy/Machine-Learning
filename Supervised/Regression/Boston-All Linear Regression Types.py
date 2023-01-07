import pandas as pd
import numpy as np
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error

################### Data Read from Sklearn Library, Dependent-Independent Variables Identified, Train-Test Split ###################
df=load_boston()
data=pd.DataFrame(df.data,columns=df.feature_names)
data2=data.copy()
data2["PRICE"]=df.target

print(data2)
y=data2["PRICE"]
X=data2.drop(columns="PRICE",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
################### Data Read from Sklearn Library, Dependent-Independent Variables Identified, Train-Test Split ###################

################### Cross Validation And Performance Functions are Created for the Values Require Returned ###################

def crossval(model):
    correctness=cross_val_score(model,X,y,cv=10)
    return correctness.mean()

def performance(reality,prediction):
    rmse=np.sqrt(mean_squared_error(reality,prediction))
    r2=mt.r2_score(reality,prediction)
    return [rmse,r2]
################### Cross Validation And Performance Functions are Created for the Values Require Returned ###################

################### Models, which want use in this dataset, Created ###################
lin_model=LinearRegression()
lin_model.fit(X_train,y_train)
lin_predict=lin_model.predict(X_test)

ridge_model=Ridge(alpha=0.1)
ridge_model.fit(X_train,y_train)
ridge_predict=ridge_model.predict(X_test)

lasso_model=Lasso(alpha=0.1)
lasso_model.fit(X_train,y_train)
lasso_predict=lasso_model.predict(X_test)

elas_model=ElasticNet(alpha=0.1)
elas_model.fit(X_train,y_train)
elas_predict=elas_model.predict(X_test)
################### Models, which want use in this dataset, Created ###################

################### Performance and Cross Validation Values for Models are Detected and Printed in DataFrame ###################
results=[["Linear Model",performance(y_test,lin_predict)[0],performance(y_test,lin_predict)[1],crossval(lin_model)],
["Ridge Model",performance(y_test,ridge_predict)[0],performance(y_test,ridge_predict)[1],crossval(ridge_model)],
["Lasso Model",performance(y_test,lasso_predict)[0],performance(y_test,lasso_predict)[1],crossval(lasso_model)],
["ElasticNet Model",performance(y_test,elas_predict)[0],performance(y_test,elas_predict)[1],crossval(elas_model)]]


results2=pd.DataFrame(results,columns=["Model","RMSE","R2","Validation"])

print(results2)
################### Performance and Cross Validation Values for Models are Detected and Printed in DataFrame ###################