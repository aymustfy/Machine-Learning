import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/diabetes.csv")
data2=data.copy()


y=data2["Outcome"]
X=data2.drop(columns="Outcome", axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=RandomForestClassifier(criterion= 'entropy', max_depth= 10, min_samples_split= 5, n_estimators= 1000, random_state=0)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

acs=accuracy_score(y_test,prediction)
print(acs*100)

###Model Tuning###

#parameters={"criterion":["gini","entropy"],
#"max_depth":[2,5,10],
#"min_samples_split":[2,5,10],
#"n_estimators":[50,200,500,1000]}

#grid=GridSearchCV(model,param_grid=parameters,cv=10,n_jobs=-1)
#grid.fit(X_train,y_train)
#print(grid.best_params_)


###verilen parametrelere bağlı olarak düşük##

plot_tree(model[0],filled=True,fontsize=5) ### index vermek zorundayız çünkü 1000 tane farklı karar ağacı çalıştı.
plt.show()

önem=pd.DataFrame({"Önem":model.feature_importances_},index=X.columns)
önem.sort_values(by="Önem", axis=0, ascending=True).plot( kind="barh", color="blue")
plt.title("Değişken Önem Seviyeleri")
plt.show()

print(önem.sum())