import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split
import sklearn.metrics as mt


################### Data Read, Correlation Checked, Dependent-Independent Variables Identified, Train-Test Split, Default Score Determined ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Ecommerce Customers.csv")
data2=data.copy()

data2.drop(["Address","Email","Avatar"],axis=1,inplace=True)

sns.heatmap(data.corr(),annot=True)
plt.show()

y=data2["Yearly Amount Spent"]
X=data2[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

elas_model=ElasticNet(random_state=2)
elas_model.fit(X_train,y_train)

predictelas=elas_model.predict(X_test)

print("\n************ElasticNet Regression(Default)************\nR2:{}\nMSE:{}\n************ElasticNet Regression(Default)************\n".format(
    elas_model.score(X_test,y_test),mt.mean_squared_error(y_test,predictelas)))
################### Data Read, Correlation Checked, Dependent-Independent Variables Identified, Train-Test Split, Default Score Determined ###################


################### Optimized Alpha Detected and Applied to ElasticNet Model ###################
lambdas=ElasticNetCV(cv=10,max_iter=10000).fit(X_train,y_train).alpha_
elas_model2=ElasticNet(alpha=lambdas)
elas_model2.fit(X_train,y_train)

predictelas2=elas_model2.predict(X_test)

print("\n************ElasticNet Regression(Optimized)************\nR2:{}\nMSE:{}\n************ElasticNet Regression(Optimized)************\n".format(
    elas_model2.score(X_test,y_test),mt.mean_squared_error(y_test,predictelas2)))
################### Optimized Alpha Detected and Applied to ElasticNet Model ###################


################### Optimized ElasticNet Model's Coefficents ###################
print("\n While the success rate of the Optimized ElasticNet Regression model is higher than default model, the MSE value is also low.\n",
"Therefore, if the variable coefficients of that model are taken into account;",
"\n Avg. Session Length---->{}\n Time on App---->{}\n Time on Website---->{}\n Length of Membership---->{}".format(
    elas_model2.coef_[0],elas_model2.coef_[1],
elas_model2.coef_[2],elas_model2.coef_[3]))
################### Optimized ElasticNet Model's Coefficents ###################
