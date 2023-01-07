import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


################### Data Read, Organised, Correlation Values Checked ###################
data=pd.read_csv("C:/Users/MSTF/Desktop/Machine Learning/Supervised/Regression/Ecommerce Customers.csv")
data2=data.copy()

data2.head()

data2.info()

data2.describe()

sns.heatmap(data2.corr(),annot=True)

data2.drop(["Address","Email","Avatar"],axis=1,inplace=True)

print(data2.head())
################### Data Read, Organised, Correlation Values Checked ###################


################### Relationship Between Independent Values and Dependent Value Checked on Graphs. ###################
sns.set_palette("flare")
sns.set_style("darkgrid")
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=data)
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=data)
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=data,kind='hex')
sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=data)
sns.pairplot(data)
plt.show()
################### Relationship Between Independent Values and Dependent Value Checked on Graphs. ###################


################### Dependent-Independent Variables Identified, Train-Test Split, Model Identify ###################
y=data2["Yearly Amount Spent"]
X=data2[["Avg. Session Length","Time on App","Time on Website","Length of Membership"]]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

Lr=LinearRegression()
Lr.fit(X_train,y_train)
predictions=Lr.predict(X_test)
################### Dependent-Independent Variables Identified, Train-Test Split, Model Identify ###################


################### Predicted Values and Y Test Values Compared on Scatter Graph. ###################
plt.scatter(y_test,predictions)
plt.xlabel("Y Test")
plt.ylabel("Predicted Y")
plt.show()
################### Predicted Values and Y Test Values Compared on Scatter Graph. ###################


################### Models Achievement Indicators Calculated ###################
r2=mt.r2_score(y_test,predictions)
mse=mt.mean_squared_error(y_test,predictions)
rmse=mt.mean_squared_error(y_test,predictions,squared=False)
mae=mt.mean_absolute_error(y_test,predictions)
print(r2,mse,rmse,mae)
sns.distplot((y_test,predictions),bins=50)
plt.show()
################### Models Achievement Indicators Calculated ###################


################### Coefficients of the Independent Variables in the Model ###################
coef= pd.DataFrame(Lr.coef_,X.columns)
coef.columns = ['Coeffecient']
print(coef)
################### Coefficients of the Independent Variables in the Model ###################


################### Coefficients are Graphically Illustrated ###################
coef2=pd.DataFrame({"Coefficents":Lr.coef_},index=X.columns)
coef2.sort_values(by="Coefficents", axis=0, ascending=True).plot(kind="bar",subplots=True)
plt.title("Coefficents of Variables")
plt.show()
################### Coefficients are Graphically Illustrated ###################


