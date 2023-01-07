import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve


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

print(data2.head())

y=data2["diagnosis"]
X=data2.drop(columns="diagnosis", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model= DecisionTreeClassifier(random_state=0,criterion="gini",
max_depth=8,max_leaf_nodes=9,min_samples_leaf=3,min_samples_split=6)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

acs=accuracy_score(y_test,prediction)
print(acs)

xisim=list(X.columns)

plot_tree(model, filled=True, fontsize=6,feature_names=xisim)
plt.show()

parameters={"criterion":["gini","entropy","log_loss"],
"max_leaf_nodes":range(2,10),
"max_depth":range(2,10),
"min_samples_split":range(2,10),
"min_samples_leaf":range(2,10)}

grid=GridSearchCV(model,param_grid=parameters,cv=10,n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)