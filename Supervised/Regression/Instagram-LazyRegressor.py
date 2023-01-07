import pandas as pd
import chardet

###################################################### Chardet used for Encoding Type Detection ######################################################
with open("C:/Users/MSTF/Desktop/Instagram.csv","rb") as x:
    result=chardet.detect(x.read())
print(result)
###################################################### Chardet used for Encoding Type Detection ######################################################

###################################################### Data Read, Preprocess ######################################################
data=pd.read_csv("C:/Users/MSTF/Desktop/Instagram.csv",encoding="Windows-1252")
data2=data.copy()
print(data2.isnull().sum())


Home=data2["From Home"].mean()
Hashtags=data2["From Hashtags"].mean()
Explore=data2["From Explore"].mean()
Other=data2["From Other"].mean()
names=("Home","Hashtags","Explore","Other")

import matplotlib.pyplot as plt

plt.pie(x=(Home,Hashtags,Explore,Other),labels=names,autopct="%.2f")
plt.show()
data2=data2.drop(columns=["From Home","From Hashtags","From Explore","From Other"],axis=1)

Impressions,Saves,Comments,Shares,Likes,PVisits,Follows,Caption,Hashtag=(data2["Impressions"],data2["Saves"],data2["Comments"],
data2["Shares"],data2["Likes"],data2["Profile Visits"],data2["Follows"],data2["Caption"],data2["Hashtags"])
listx=[Saves,Shares,Likes,PVisits,Follows,Comments]

data2=data2.drop(columns=["Caption","Hashtags"],axis=1)

import seaborn as sns
for i in listx:
    sns.regplot(x=i,y="Impressions", ci=None,data=data2,color="r")
    plt.show()
###################################################### Data Read, Preprocess ######################################################

############################# Train-Test Split, Model Detect with Lazy Regressor by R-Squared (R_S=2) #############################
y=Impressions

from sklearn.model_selection import train_test_split,GridSearchCV
from lazypredict.Supervised import LazyRegressor

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
model=LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models,prediction=model.fit(X_train,X_test,y_train,y_test)
sorting=models.sort_values(by="R-Squared",ascending=False)
print(sorting)
############################# Train-Test Split, Model Detect with Lazy Regressor by R-Squared (R_S=2) #############################

######################### Models Imported, Parameters Identified and Tried the Optimize with Best Params #########################
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor,ExtraTreesRegressor
import sklearn.metrics as mt
models=[RandomForestRegressor(random_state=2),ExtraTreesRegressor(random_state=2),
BaggingRegressor(random_state=2),DecisionTreeRegressor(random_state=2)]

rfparameters={"max_depth":range(0,20),"max_features":range(0,20),"n_estimators":range(0,20)}
etparameters={"max_depth":range(0,20),"min_samples_split":range(0,20),"min_samples_leaf":(1,10)}
bgparameters={"n_estimators":range(0,20),"max_features":range(0,20),"max_samples":range(0,20)}
dtparameters={"min_samples_split":range(0,20),"max_leaf_nodes":range(0,20)}
parameters=[rfparameters,etparameters,bgparameters,dtparameters]

for i in range(0,4):
    Modelnames=["Rf","Et","Bg","Dt"]
    grid=GridSearchCV(estimator=models[2],param_grid=parameters[2],cv=10)
    grid.fit(X_train,y_train)
    print(Modelnames[i],"\n",grid.best_params_,"\n\n")
######################### Models Imported, Parameters Identified and Tried the Optimize with Best Params #########################

# Models Created, Function Identified for R2 and RMSE, DataFrame Created with Results, Feature Importances to List for XGB Model Which Have Best Score #
models=[XGBRegressor(objective='reg:squarederror'), RandomForestRegressor(random_state=2,max_depth=11,max_features=5,n_estimators=45,min_samples_split=2,
max_leaf_nodes=20),
ExtraTreesRegressor(random_state=2,max_features=5,n_estimators=15, min_samples_leaf=10,min_samples_split=20),
BaggingRegressor(random_state=1,max_features=5,n_estimators=45,max_samples=75,bootstrap_features=True,n_jobs=35),
DecisionTreeRegressor(random_state=2,max_leaf_nodes=44,min_samples_split=13,max_depth=45,max_features=5)]

def modelprediction(model):
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    r2=mt.r2_score(y_test,prediction)
    rmse=mt.mean_squared_error(y_test,prediction,squared=False)
    return [r2,rmse]

result=[]
for i in models:
    result.append(modelprediction(i))

Names=["XGB Model","Random Forest Model","Extra Tree Model","Bagging Model","Decision Tree Model"]
df=pd.DataFrame(Names,columns=["Model Name"])
df2=pd.DataFrame(result,columns=["R2","RMSE"])
df=df.join(df2)
print(df.sort_values(by="R2",ascending=False))

xgb=XGBRegressor(models[0])
lisst=xgb.feature_importances_

features=["Saves","Comments","Shares","Likes","PVisits","Follows"]
print(list(zip(lisst,features)))
## Models Created, Function Identified for R2 and RMSE, DataFrame Created with Results, Feature Importances to List for XGB Model Which Have Best Score ##

############################# Train-Test Split, Model Detect with Lazy Regressor by R-Squared (R_S=42) #############################
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
models,prediction=model.fit(X_train,X_test,y_train,y_test)
sorting=models.sort_values(by="R-Squared",ascending=False)
print(sorting)
############################# Train-Test Split, Model Detect with Lazy Regressor by R-Squared (R_S=42) #############################

###################### Models Imported, RANSAC Regression Selected as Best Model, Coefficients are Written about RANSAC Regression ######################
from sklearn.linear_model import RANSACRegressor,OrthogonalMatchingPursuitCV,HuberRegressor


models2=[RANSACRegressor(random_state=42), OrthogonalMatchingPursuitCV(),
XGBRegressor(objective='reg:squarederror'),HuberRegressor()]

def modelprediction2(model):
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    r2=mt.r2_score(y_test,prediction)
    rmse=mt.mean_squared_error(y_test,prediction,squared=False)
    return [r2,rmse]

result2=[]
for i in models2:
    result2.append(modelprediction2(i))

Names2=["RANSAC","OrthogonalMPCV","XGB","Huber"]
df3=pd.DataFrame(Names2,columns=["Model Name"])
df4=pd.DataFrame(result2,columns=["R2","RMSE"])
df3=df3.join(df4)
print(df3.sort_values(by="R2",ascending=False))

from sklearn import linear_model
import numpy as np
from sklearn import linear_model

ransac=linear_model.RANSACRegressor(random_state=42)
ransac.fit(X_train,y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
print(ransac.estimator_.coef_,ransac.estimator_.intercept_,mt.r2_score(y_test,ransac.predict(X_test)))

coef=ransac.estimator_.coef_.tolist()
list2=[]

for i in range(0,6):
    list2.append("{:.2f}".format(coef[i]))
    
print("********************************Model Features Coefficents:********************************\n",list(zip(features,list2)))
###################### Models Imported, RANSAC Regression Selected as Best Model, Coefficients are Written about RANSAC Regression ######################
