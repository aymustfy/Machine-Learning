import pandas as pd
import numpy as np
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor


data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Advertising.csv")
data2=data.copy()


y=data2["Sales"]
X=data2.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


############### default parameters 
dtmodel=DecisionTreeRegressor(random_state=0)
dtmodel.fit(X_train,y_train)
dtpredict=dtmodel.predict(X_test)

r2=mt.r2_score(y_test,dtpredict)
rmse=mt.mean_squared_error(y_test,dtpredict,squared=False)

bgmodel=BaggingRegressor(random_state=0)
bgmodel.fit(X_train,y_train)
bgpredict=bgmodel.predict(X_test)

r2bg=mt.r2_score(y_test,bgpredict)
rmsebg=mt.mean_squared_error(y_test,bgpredict,squared=False)

rfmodel=RandomForestRegressor(random_state=0)
rfmodel.fit(X_train,y_train)
rfpredict=rfmodel.predict(X_test)

r2rf=mt.r2_score(y_test,rfpredict)
rmserf=mt.mean_squared_error(y_test,rfpredict,squared=False)

print("\n*****************Default Model Scores*****************\n","Decision Tree R2:{}       Decision Tree RMSE:{}\n Bagging R2:{}       Bagging RMSE:{}\n Random Forest R2:{}       Random Forest RMSE:{}".format(
    r2,rmse,r2bg,rmsebg,r2rf,rmserf),"\n*****************Default Model Scores*****************\n\n")

############### default parameters


############### parameter optimization
parametersdt={"min_samples_split":range(2,25),"max_leaf_nodes":range(2,25)}
grid1=GridSearchCV(estimator=dtmodel,param_grid=parametersdt,cv=10)
grid1.fit(X_train,y_train)

parametersbg={"n_estimators":range(2,25)}
grid2=GridSearchCV(estimator=bgmodel,param_grid=parametersbg,cv=10)
grid2.fit(X_train,y_train)

parametersrf={"max_depth":range(2,20),"max_features":range(2,20),"n_estimators":range(2,20)}
grid3=GridSearchCV(estimator=rfmodel,param_grid=parametersrf,cv=10)
grid3.fit(X_train,y_train)

print("\n*****************Optimize Parameters*****************\n","Decision Tree Parameters:{}       \n Bagging Parameters:{}  \n Random Forest Parameters:{}".format(
    grid1.best_params_,grid2.best_params_,grid3.best_params_),"\n*****************Default Model Scores*****************\n\n")
############### parameter optimization


############### optimized parameters in model
dtmodel=DecisionTreeRegressor(random_state=0,max_leaf_nodes=18,min_samples_split=4)
dtmodel.fit(X_train,y_train)
dtpredict=dtmodel.predict(X_test)

r2=mt.r2_score(y_test,dtpredict)
rmse=mt.mean_squared_error(y_test,dtpredict,squared=False)

feature_importances_dt = np.mean([
    tree.feature_importances_ for tree in bgmodel.estimators_
], axis=0)

bgmodel=BaggingRegressor(random_state=0,n_estimators=23)
bgmodel.fit(X_train,y_train)
bgpredict=bgmodel.predict(X_test)

r2bg=mt.r2_score(y_test,bgpredict)
rmsebg=mt.mean_squared_error(y_test,bgpredict,squared=False)

feature_importances_bg = np.mean([
    tree.feature_importances_ for tree in bgmodel.estimators_
], axis=0)

rfmodel=RandomForestRegressor(random_state=0,n_estimators=16,max_depth=8,max_features=2)
rfmodel.fit(X_train,y_train)
rfpredict=rfmodel.predict(X_test)

r2rf=mt.r2_score(y_test,rfpredict)
rmserf=mt.mean_squared_error(y_test,rfpredict,squared=False)
print("*****************Optimized Scores*****************\n","Decision Tree R2:{}       Decision Tree RMSE:{}\n Bagging R2:{}       Bagging RMSE:{}\n Random Forest R2:{}       Random Forest RMSE:{}".format(
    r2,rmse,r2bg,rmsebg,r2rf,rmserf),"\n*****************Optimized Scores*****************\n\n")

feature_importances_rf = np.mean([
    tree.feature_importances_ for tree in rfmodel.estimators_
], axis=0)
print("*****************Optimized Feature Importances*****************\n","Decision Tree----->:{}\n Bagging----->:{}\n Random Forest----->:{}".format(
    feature_importances_dt,feature_importances_bg,feature_importances_rf),"\n*****************Optimized Feature Importances*****************")
############### optimized parameters in model