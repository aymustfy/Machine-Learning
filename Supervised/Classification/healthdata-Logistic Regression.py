import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Logistic Regression/health_data.csv")
data2=data.copy()


Diabetes=data2["Diabetes"]

Hypertension=data2["Hypertension"]

Stroke=data2["Stroke"]

X=data2.drop(columns=["Diabetes","Hypertension","Stroke"],axis=1)
list3=(0,1,2)

for i in list3:
    listd=(Diabetes,Hypertension,Stroke)
    list2=["Diabetes","Hypertension","Stroke"]
    
    X_train,X_test,y_train,y_test=train_test_split(X,listd[i],test_size=0.2,random_state=42)
    
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    
    model=LogisticRegression(random_state=42)
    model.fit(X_train,y_train)
    prediction=model.predict(X_test)

    cm=confusion_matrix(y_test,prediction)

    acs=accuracy_score(y_test,prediction)
    
    cr=classification_report(y_test,prediction)

    auc=roc_auc_score(y_test,prediction)

    fpr,tpr,thresold=roc_curve(y_test,model.predict_proba(X_test)[:,1])

    plt.plot(fpr,tpr,label="Model AUC(Alan=%0.2f)"%auc)
    plt.plot([0,1],[0,1],"r--")
    plt.xlabel("False Positive")
    plt.ylabel("True Positive")
    plt.title("ROC")
    plt.legend(loc="lower right")
    plt.show()
    list2=["Diabetes","Hypertension","Stroke"]

    
    print("\n************************{}************************\n\nConfusion Matrix= \n {}          \n\nAccuracy Score= {}           \n\nClassification Report= \n{}         \n\nRoc Auc Score= {}\n\n***************Coefficents***************".format(list2[i],cm,acs,cr,auc))
    print(model.coef_,"\n************************{}************************\n".format(list2[i]))