import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Train-Test Sets Standardizated ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Classification/winequality.csv")
data2=data.copy()


y=data["quality"]
X=data.drop(columns="quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=(sc.fit_transform(X_train))
X_test=sc.transform(X_test)

pca=PCA(n_components=3,random_state=1)
X_train2=pca.fit_transform(X_train)
X_test2=pca.fit_transform(X_test)
print(np.cumsum(pca.explained_variance_ratio_)*100)
################### Data Read, Dependent-Independent Variables Identified, Train-Test Split, Train-Test Sets Standardizated, PCA Model Created ###################

################### Linear Discriminant Analysis Model Created, Variance Ratios for LDA Calculated ###################
lda=LinearDiscriminantAnalysis(n_components=4)
X_train3=lda.fit_transform(X_train,y_train)
X_test3=lda.transform(X_test)

print(np.cumsum(lda.explained_variance_ratio_)*100)
################### Linear Discriminant Analysis Model Created, Variance Ratios for LDA Calculated ###################

################### Linear Regression Model Created, R2 and RMSE Values Calculated ###################
lm=LinearRegression()
lm.fit(X_train3,y_train)
prediction=lm.predict(X_test3)
r2=mt.r2_score(y_test,prediction)
rmse=mt.mean_squared_error(y_test,prediction,squared=True)

print("R2:{}     RMSE:{}".format(r2,rmse))

cv=KFold(n_splits=10,shuffle=True,random_state=1)
lm2=LinearRegression()
RMSE=[]

for i in range(1,X_train2.shape[1]+1):
    error=np.sqrt(-1*cross_val_score(lm2,X_train2[:,:i],y_train.ravel(),cv=cv,scoring="neg_mean_squared_error").mean())
    RMSE.append(error)

plt.plot(RMSE,"-x")
plt.xlabel("count of components")
plt.ylabel("RMSE")
plt.show()

