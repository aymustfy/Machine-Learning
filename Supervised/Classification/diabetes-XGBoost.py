import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/diabetes.csv")
data2=data.copy()


y=data2["Outcome"]
X=data2.drop(columns="Outcome", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

modelxgb=XGBClassifier(learning_rate=0.2,max_depth=3,n_estimators=500,subsample=0.7)
modelxgb.fit(X_train,y_train)
prediction=modelxgb.predict(X_test)

acs=accuracy_score(y_test,prediction)
print(acs)

parameters={"max_depth":[3,5,7],"subsample":[0.2,0.5,0.7],
"n_estimators":[500,1000,2000],"learning_rate":[0.2,0.5,0.7]}

grid=GridSearchCV(modelxgb,param_grid=parameters,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)

###Burada model başarısı, parametrelerin değişmesine 
# göre yükselir ancak işlem uzun sürmektedir.
