import pandas as pd
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor,RandomForestRegressor


data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Advertising.csv")
data2=data.copy()

y=data2["Sales"]
X=data2.drop(columns="Sales",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

def modelpredict(model):
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    r2=mt.r2_score(y_test,prediction)
    rmse=mt.mean_squared_error(y_test,prediction,squared=False)
    return(r2,rmse)

models=[LinearRegression(),Ridge(),Lasso(),ElasticNet(),SVR(),DecisionTreeRegressor(random_state=0),
BaggingRegressor(random_state=0),RandomForestRegressor(random_state=0)]

result=[]
for i in models:
    result.append(modelpredict(i))

modelsname=["Linear Model","Ridge Model","Lasso Model","ElasticNet Model","SVR Model","Decision Tree Model","Bagging Model","Random Forest Model"] 

df=pd.DataFrame(modelsname,columns=["Model Name"])
df2=pd.DataFrame(result, columns=["R2","RMSE"])

df=df.join(df2)

print(df)


#### Bagging and Random Forest models give us the best results when no optimization is applied.