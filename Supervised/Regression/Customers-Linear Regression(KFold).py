import pandas as pd
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold


################### Data Read, Organised, Dependent-Independent Variables Identified, Train-Test Split, Model Identify ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Ecommerce Customers.csv")


data.drop(["Address","Email","Avatar"],axis=1,inplace=True)

y=data["Yearly Amount Spent"]
X=data[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Lr=LinearRegression()
Lr.fit(X_train,y_train)
predictions=Lr.predict(X_test)
################### Data Read, Organised, Dependent-Independent Variables Identified, Train-Test Split, Model Identify ###################


########################### Fuction Identify ###########################
def score(model,x_train,x_test,y_train,y_test):
    trainpredict=model.predict(x_train)
    testpredict=model.predict(x_test)

    r2_train=mt.r2_score(y_train,trainpredict)
    r2_test=mt.r2_score(y_test,testpredict)

    mse_train=mt.mean_squared_error(y_train,trainpredict)
    mse_test=mt.mean_squared_error(y_test,testpredict)

    coef=model.coef_
    
    return[r2_train,r2_test,mse_train,mse_test,coef]
########################### Fuction identified. ###########################


########################### Model Applied by Default Parameters ###########################
result=score(model=Lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
print("\n\n****************************< Default Model >****************************:\n")
print("Train R2={}      Train MSE={}".format(result[0],result[2]))
print("Test R2={}      Test MSE={}".format(result[1],result[3]))
print("Coefficients:{}".format(result[4]))
########################### Model Applied by Default Parameters ###########################


########################### Model Optimized ###########################
lr_cv=LinearRegression()
k=5
iteration=1
cv=KFold(n_splits=k)

for trainindex,testindex in cv.split(X):
    X_train,X_test=X.loc[trainindex], X.loc[testindex]
    y_train,y_test=y.loc[trainindex], y.loc[testindex]

    lr_cv.fit(X_train,y_train)

    result2=score(model=lr_cv,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
  
    print("\n\n****************************< iteration---{} >****************************:\n".format(iteration))
    print("Train R2={}      Train MSE={}".format(result2[0],result2[2]))
    print("Test R2={}      Test MSE={}".format(result2[1],result2[3]))
    print("Coefficients:{}".format(result2[4]))
    
    iteration+=1
########################### Model Optimized ###########################





#### Although R2 and MSE values were similar between iterations, iteration 3 was chosen. Accordingly;

#### Yearly Amount Spent  =  25.649*Avg.Session Lenght  +  38.798*Time on App  +  0.363*Time on Website  +  61.665*Lenghth of Membership
