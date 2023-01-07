import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm
from sklearn.metrics import accuracy_score


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/KNN/cancer.csv")
data2=data.copy()

data2=data2.drop(columns=["id"],axis=1)

M=data2[data2["diagnosis"]=="M"]
B=data2[data2["diagnosis"]=="B"]

plt.scatter(M.radius_mean, M.texture_mean, color="red",label="ill-natured")
plt.scatter(B.radius_mean, B.texture_mean, color="green",label="well-natured")
plt.legend()
plt.show()


data2.diagnosis=[1 if kod=="M" else 0 for kod in data2.diagnosis]


y=data2["diagnosis"]
X=data2.drop(columns="diagnosis", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=GaussianNB()
model.fit(X_train,y_train)
prediction=model.predict(X_test)

acs=accuracy_score(y_test,prediction)
print(acs*100)

