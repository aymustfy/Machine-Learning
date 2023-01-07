import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Train-Test Sets Standardizated ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Classification/winequality.csv")
data2=data.copy()

y=data2["quality"]
X=data2.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=(sc.fit_transform(X_train))
X_test=sc.transform(X_test)
################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Train-Test Sets Standardizated ###################

################### Principal Component Analysis Model Created, Variance Ratios for PCA Calculated ###################
pca=PCA(random_state=1)
X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)

print(np.cumsum(pca.explained_variance_ratio_)*100)
################### Principal Component Analysis Model Created, Variance Ratios for PCA Calculated ###################

################### Linear Regression Model Created, R2 and RMSE Values Calculated ###################
lm=LinearRegression()
lm.fit(X_train2,y_train)
prediction=lm.predict(X_test2)
r2=mt.r2_score(y_test,prediction)
rmse=mt.mean_squared_error(y_test,prediction,squared=True)

print("\n************For All Components************\nR2:{}     RMSE:{}\n************For All Components************".format(r2,rmse))
################### Linear Regression Model Created, R2 and RMSE Values Calculated ###################

################### K-fold is applied to the model,Ideal Number of Components are Detected by Graphic ###################
cv=KFold(n_splits=10,shuffle=True,random_state=1)
lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    error=np.sqrt(-1*cross_val_score(lm2,X_train2[:,:i],y_train.ravel(),
    cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(error)

plt.plot(RMSE,"-x")
plt.xlabel("Number of Components")
plt.ylabel("RMSE")
plt.show()
################### K-fold is applied to the model,Ideal Number of Components are Detected by Graphic ###################

################### Ideal Number of Components are Applied the Model and  ###################
pca=PCA(n_components=3, random_state=1)
X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)

lm2=LinearRegression()
lm2.fit(X_train2,y_train)
prediction2=lm2.predict(X_test2)
r2=mt.r2_score(y_test,prediction2)
rmse=mt.mean_squared_error(y_test,prediction2,squared=True)

print("\n************For Ideal Number of Components(3)************\nR2:{}     RMSE:{}\n************For Ideal Number of Components(3)************".format(r2,rmse))
################### Ideal Number of Components are Applied the Model ###################

print("The Analysis model chosen for this dataset is most likely incorrect. R2 and RMSE values shows that. But this model just simple example for PCA.",
"If more suitable models apply for dataset, these values could optimize.")