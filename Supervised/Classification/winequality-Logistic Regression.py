import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score


data=pd.read_csv("C:/Users/MSTF/Desktop/Git/Logistic Regression/winequality.csv")
data2=data.copy()

print(data2["quality"].unique())

category=["3","4","5","6","7","8"]
oe=OrdinalEncoder(categories=[category])
data2["Quality"]=oe.fit_transform(data2["quality"].values.reshape(-1,1))
data2=data2.drop(columns="quality",axis=1)

sns.heatmap(data2.corr(),annot=True)
plt.show()


y=data2["Quality"]
X=data2.drop(columns="Quality",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

model=LogisticRegression(random_state=10,max_iter=1224)
model.fit(X_train,y_train)
prediction=model.predict(X_test)

cm=confusion_matrix(y_test,prediction)
print(cm)

acs=accuracy_score(y_test,prediction)
print(acs)
