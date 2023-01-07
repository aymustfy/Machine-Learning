import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/diabetes.csv")
data2=data.copy()


y=data2["Outcome"]
X=data2.drop(columns="Outcome", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

def models(model):
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)
    score=accuracy_score(y_test,prediction)
    return round(score*100,2)


print(models(DecisionTreeClassifier()))

models2=[]
models2.append(("Log Regression",LogisticRegression(random_state=0)))
models2.append(("K Neighbors",KNeighborsClassifier()))
models2.append(("SVC",SVC(random_state=0)))
models2.append(("Bayes",GaussianNB()))
models2.append(("Decision Tree",DecisionTreeClassifier(random_state=0)))

print(models2)

modelname=[]
success=[]

for i in models2:
    modelname.append(i[0])
    success.append(models(i[1]))

print(success)

a=list(zip(modelname,success))
result=pd.DataFrame(a,columns=["Model","Score"])
print(result)

#need optimization(hiperparameters)