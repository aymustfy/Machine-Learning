import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Logistic Regression/adm_data.csv")
data2=data.copy()

data2=data2.drop(columns="Serial No.",axis=1)

data2["Chance of Admit"]=pd.to_numeric(data2["Chance of Admit "],errors="coerce") 

data2=data2.drop(columns=["Chance of Admit "],axis=1)



y=data2["Chance of Admit"]

X=data2.drop(columns=["Chance of Admit"],axis=1)

print(data2)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


model=RandomForestClassifier(random_state=42)
model2=model.fit(X_train,y_train)
prediction=model2.predict(X_test)

cm=confusion_matrix(y_test,prediction)

acs=accuracy_score(y_test,prediction)
    
cr=classification_report(y_test,prediction)

auc=roc_auc_score(y_test,prediction)

print("Confusion Matrix= \n {}          \nAccuracy Score= {}           \nClassification Report= \n{}         \nRoc Auc Score= {}".format(cm,acs,cr,auc))


