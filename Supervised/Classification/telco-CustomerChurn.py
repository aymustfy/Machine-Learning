import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC,SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Telco.csv")
data2=data.copy()

#print(data2.shape)

#print(data2.info())

data2=data2.drop(columns="customerID",axis=1)



###Türkçeleştirme ekle....

data2["TotalCharges"]=pd.to_numeric(data2["TotalCharges"],errors="coerce")

#print(data2.info())

#print(data2.isnull().sum())

#print(data2[["tenure","MonthlyCharges","TotalCharges"]])

#data2["Difference"]=data2["TotalCharges"]-(data2["tenure"]*data2["MonthlyCharges"])
#print(data2["Difference"])
#print(data2["Difference"].max())

#print(data2[data2["tenure"]==0])

#print(data2[data2["MonthlyCharges"]==0])

data2=data2.dropna()
#print(data2.isnull().sum())

#print(data2.describe())

#print(data2.select_dtypes(include="object").columns)

le=LabelEncoder()
variable=data2.select_dtypes(include="object").columns
data2.update(data2[variable].apply(le.fit_transform))
#print(data.head())
print(data2.info())


data2["Churn"]=pd.to_numeric(data2["Churn"],errors="coerce")

y=data2["Churn"]
X=data2.drop(columns="Churn",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#clf=LazyClassifier()
#models,prediction=clf.fit(X_train,X_test,y_train,y_test)
#sorting=models.sort_values(by="Accuracy",ascending=True)

#plt.barh(sorting.index,sorting["Accuracy"])
#plt.show()


modelsname=["LinearSVC","SVC","Ridge","Logistic","RandomForest","LGBM","XGBM"]
models=[LinearSVC(random_state=0,C=0.1,penalty="l2",dual=False),SVC(random_state=0,C=0.1,gamma=0.01,kernel="linear"),
RidgeClassifier(random_state=0,alpha=1.0),LogisticRegression(random_state=0,C=1,penalty="l2",dual=False),
RandomForestClassifier(random_state=0,max_depth=10,min_samples_split=5,n_estimators=2000),
LGBMClassifier(random_state=0,learning_rate=0.01,max_depth=4,n_estimators=1000,subsample=0.6),
XGBClassifier(learning_rate=0.001,max_depth=4,n_estimators=2000,subsample=0.6)]

parameters={
    modelsname[0]:{"C":[0.1,1,10,100],"penalty":["l1","l2"]},
    modelsname[1]:{"kernel":["linear","rbf"],"C":[0.1,1],"gamma":[0.01,0.001]},
    modelsname[2]:{"alpha":[0.1,1.0]},
    modelsname[3]:{"C":[0.1,1],"penalty":["l1","l2"]},
    modelsname[4]:{"n_estimators":[1000,2000],"max_depth":[4,10],"min_samples_split":[2,5]},
    modelsname[5]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],
    "subsample":[0.6,0.8]},
    modelsname[6]:{"learning_rate":[0.001,0.01],"n_estimators":[1000,2000],"max_depth":[4,10],
    "subsample":[0.6,0.8]}}
#print(parameters["Ridge"])

def result(model):
    model.fit(X_train,y_train)
    return model

def score(model2):
    prediction=result(model2).predict(X_test)
    acs=accuracy_score(y_test,prediction)
    return acs*100

#print(score(models[0]))

#m=LinearSVC(random_state=0)
#m.fit(X_train,y_train)
#t=m.predict(X_test)
#s=accuracy_score(y_test,t)
#print(s*100)

#for i,j in zip(modelsname,models):
#    print(i)
 #   grid=GridSearchCV(result(j),param_grid=parameters[i],cv=10,n_jobs=-1)
  #  grid.fit(X_train,y_train)
   # print(grid.best_params_)

achieve=[]
for i in models:
    achieve.append(score(i))
print(achieve)


a=list(zip(modelsname,achieve))
results=pd.DataFrame(a,columns=["Model","Achieve"])
print(results.sort_values("Achieve",ascending=False))



