import pandas as pd
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score


################### Data Read, Dependent-Independent Variables Identified, Train-Test Split ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Ecommerce Customers.csv")
data2=data.copy()

y=data2["Yearly Amount Spent"]
X=data2[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
################### Data Read, Dependent-Independent Variables Identified, Train-Test Split ###################


################### Cross Validation and Models Success Parameters Determined as a Function ###################
def crossval(model):
    correctness=cross_val_score(model,X,y,cv=10)
    return correctness.mean()

def performance(reality,prediction):
    rmse=mt.mean_squared_error(reality,prediction,squared=True)
    r2=mt.r2_score(reality,prediction)
    return [rmse,r2]
################### Cross Validation and Models Success Parameters Determined as a Function ###################


################### Models Determined ###################
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
################### Models Determined ###################


################### Models Performance and Cross Validation Scores Defined and then Printed as DataFrame ###################
results=[["Linear Model",performance(y_test,lin_predict)[0],performance(y_test,lin_predict)[1],crossval(lin_model)],
["Ridge Model",performance(y_test,ridge_predict)[0],performance(y_test,ridge_predict)[1],crossval(ridge_model)],
["Lasso Model",performance(y_test,lasso_predict)[0],performance(y_test,lasso_predict)[1],crossval(lasso_model)],
["ElasticNet Model",performance(y_test,elas_predict)[0],performance(y_test,elas_predict)[1],crossval(elas_model)]]


results2=pd.DataFrame(results,columns=["Model","RMSE","R2","Validation"])

results3=[["Linear Model",lin_model.coef_[0],lin_model.coef_[1],lin_model.coef_[2],lin_model.coef_[3]],
["Ridge Model",ridge_model.coef_[0],ridge_model.coef_[1],ridge_model.coef_[2],ridge_model.coef_[3]],
["Lasso Model",lasso_model.coef_[0],lasso_model.coef_[1],lasso_model.coef_[2],lasso_model.coef_[3]],
["ElasticNet Model",elas_model.coef_[0],elas_model.coef_[1],elas_model.coef_[2],elas_model.coef_[3]]]

results4=pd.DataFrame(results3,columns=["Model Name","Avg. Session Length","Time on App","Time on Website","Length of Membership"])
print(results2,"\n")
print("\n",results4)
################### Models Performance and Cross Validation Scores Defined and then Printed as DataFrame ###################

print("\n\nBest model for this situation seems to be Lasso regression. Accordingly independent variable coefficients;\nAvg.Session Length---->25.49\nTime on App---->38.69\nTime on Website---->0.22\nLength of Membership---->61.81")
