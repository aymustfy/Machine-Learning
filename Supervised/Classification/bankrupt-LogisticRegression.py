import pandas as pd
import chardet

with open("C:/Users/MSTF/Desktop/Bankrupt.csv","rb") as x:
    result=chardet.detect(x.read())
print(result)

data=pd.read_csv("C:/Users/MSTF/Desktop/Bankrupt.csv",encoding="ISO-8859-1")
data2=data.copy()
data2=data2.dropna(axis=0)
data2.rename(columns={"Bankrupt?":"Bankrupt"},inplace=True)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


y=data2["Bankrupt"]
X=data2.drop(columns="Bankrupt")

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


model=LogisticRegression(random_state=42)
model.fit(X_train,y_train)
predict=model.predict(X_test)

import sklearn.metrics as mt 
r2=mt.accuracy_score(predict,y_test)
rmse=mt.mean_squared_error(predict,y_test,squared=False)
print("\n\nModel Score:",r2,"\nModel Root Mean Squared Error",rmse)