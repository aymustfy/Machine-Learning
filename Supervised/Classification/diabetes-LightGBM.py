import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/diabetes.csv")
data2=data.copy()

y=data2["Outcome"]
X=data2.drop(columns="Outcome", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LGBMClassifier(learning_rate=0.01,max_depth=3,n_estimators=1000,subsample=0.6)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

acs=accuracy_score(y_test,prediction)
print(acs)

parameters={"max_depth":[3,5,7],"subsample":[0.6,0.8,1.0],
"n_estimators":[200,500,1000],"learning_rate":[0.001,0.01,0.1]}

grid=GridSearchCV(model,param_grid=parameters,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)