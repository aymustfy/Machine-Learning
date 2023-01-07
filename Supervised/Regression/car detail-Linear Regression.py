import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sklearn.metrics as mt



data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Linear Regression/car detail.csv")

print(data.duplicated().sum()) #veri setinde tekrarlanan veriler mevcut

data.drop_duplicates(inplace=True) #veri setinde tekrarlanan veriler mevcut ve bu veriler yapılacak analizde yanlış sonuçlara sebebiyet verebilecektir.

data.reset_index(inplace=True) #veri indeksleri yeniden düzenlendi.

data.drop(columns="index", inplace=True) #index ataması sonrası önceki index verilerinin tutulduğu kolonlar silindi.

#print(data)

print('\n Fuel Types  - ',data['fuel'].unique())
print('\n Seller Types  - ',data['seller_type'].unique())
print('\n Transmission types - ',data['transmission'].unique())
print('\n Owner types - ',data['owner'].unique())
#dummy atayacağımız kolonlarda uniq veri sayılarını bulduk.


Fuel_type=pd.get_dummies(data['fuel'],drop_first=True)
Seller_Type=pd.get_dummies(data['seller_type'],drop_first=True)
Transmission=pd.get_dummies(data['transmission'],drop_first=True)
Owner=pd.get_dummies(data['owner'],drop_first=True)
#dummy ataması yaptık.


data=data.drop(['fuel','seller_type','transmission','name',"owner"],axis=1)
data=pd.concat([data,Fuel_type,Seller_Type,Transmission,Owner],axis=1)
print(data)
#stringleri sildik ve dummyleri veriye entegre ettik

y=data["selling_price"]
X=data.drop(columns="selling_price")
#bağımlı ve bağımsız değişkenleri ayırdık.

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#train test gruplarını ayırdık.

Lr=LinearRegression()
Lr.fit(X_train,y_train)
predictions=Lr.predict(X_test)
#Linear regression modeli çalıştık.

r2=mt.r2_score(y_test,predictions)
mse=mt.mean_squared_error(y_test,predictions)
rmse=mt.mean_squared_error(y_test,predictions,squared=False)
mae=mt.mean_absolute_error(y_test,predictions)
#model kontrolü için gerekli bileşenleri bulduk.

coef= pd.DataFrame(Lr.coef_,X.columns)
coef.columns = ['Coeffecient']
print(coef)
#modeldeki bağımsız değişken katsayılarını bulduk.

print(r2,mse,rmse,mae) #marka ve isim, araç fiyatları üzerinde oldukça etkili.

