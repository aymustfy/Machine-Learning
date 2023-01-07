import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
import statsmodels.api as sm
import sklearn.metrics as mt



data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Linear Regression/car detail.csv")

print(data.duplicated().sum()) #veri setinde tekrarlanan veriler mevcut

data.drop_duplicates(inplace=True) #veri setinde tekrarlanan veriler mevcut ve bu veriler yapılacak analizde yanlış sonuçlara sebebiyet verebilecektir.

data.reset_index(inplace=True) #veri indeksleri yeniden düzenlendi.

data.drop(columns="index", inplace=True) #index ataması sonrası önceki index verilerinin tutulduğu kolonlar silindi.

#print(data)

#print('\n Fuel Types  - ',data['fuel'].unique())
#print('\n Seller Types  - ',data['seller_type'].unique())
#print('\n Transmission types - ',data['transmission'].unique())
#print('\n Owner types - ',data['owner'].unique())
#dummy atayacağımız kolonlarda uniq veri sayılarını bulduk.


Fuel_type=pd.get_dummies(data['fuel'],drop_first=True)
Seller_Type=pd.get_dummies(data['seller_type'],drop_first=True)
Transmission=pd.get_dummies(data['transmission'],drop_first=True)
Owner=pd.get_dummies(data['owner'],drop_first=True)
#dummy ataması yaptık.


data=data.drop(['fuel','seller_type','transmission','name',"owner"],axis=1)
data=pd.concat([data,Fuel_type,Seller_Type,Transmission,Owner],axis=1)
#print(data)
#stringleri sildik ve dummyleri veriye entegre ettik

y=data["selling_price"]
X=data.drop(columns="selling_price")
#bağımlı ve bağımsız değişkenleri ayırdık.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#train test gruplarını ayırdık.

Lr=LinearRegression()
model=Lr.fit(X_train,y_train)
predictions=Lr.predict(X_test)
#Linear regression modeli çalıştık.

def score(model,x_train,x_test,y_train,y_test):
    trainpredict=model.predict(x_train)
    testpredict=model.predict(x_test)

    r2_train=mt.r2_score(y_train,trainpredict)
    r2_test=mt.r2_score(y_test,testpredict)

    mse_train=mt.mean_squared_error(y_train,trainpredict)
    mse_test=mt.mean_squared_error(y_test,testpredict)

    return[r2_train,r2_test,mse_train,mse_test]

result=score(model=Lr,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
print("Train R2={}      Train MSE={}".format(result[0],result[2]))
print("Test R2={}      Test MSE={}".format(result[1],result[3]))


lr_cv=LinearRegression()
k=5
iteration=1
cv=KFold(n_splits=k)

for trainindex,testindex in cv.split(X):
    X_train,X_test=X.loc[trainindex], X.loc[testindex]
    y_train,y_test=y.loc[trainindex], y.loc[testindex]

    lr_cv.fit(X_train,y_train)

    result2=score(model=lr_cv,x_train=X_train,x_test=X_test,y_train=y_train,y_test=y_test)
    
    print("iteration:{}".format(iteration))
    print("Train R2={}      Train MSE={}".format(result2[0],result2[2]))
    print("Test R2={}      Test MSE={}".format(result2[1],result2[3]))

    iteration+=1

