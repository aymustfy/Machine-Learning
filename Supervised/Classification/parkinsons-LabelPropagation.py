import pandas as pd

data=pd.read_csv("C:/Users/MSTF/Desktop/parkinsons.data")
data2=data.copy()


data2=data2.drop(columns="name")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X=sc.fit_transform(data2.drop(columns="status"))
y=data2["status"]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

from lazypredict.Supervised import LazyClassifier
lc=LazyClassifier(random_state=42)
models,predict=lc.fit(X_train,X_test,y_train,y_test)
sorting=models.sort_values(by="Accuracy",ascending=False)
print(sorting)

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.semi_supervised import LabelPropagation

lp=LabelPropagation()
lp.fit(X_train,y_train)
y_predict=lp.predict(X_test)

acs=accuracy_score(y_test,y_predict)
cm=confusion_matrix(y_test,y_predict)
print(acs,"\n",cm)