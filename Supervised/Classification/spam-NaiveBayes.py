from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import chardet

with open("C:/Users/MSTF/Desktop/Git/spam.csv","rb") as x:
    result=chardet.detect(x.read())
print(result)

data=pd.read_csv("C:/Users/MSTF/Desktop/Git/spam.csv",encoding="Windows-1252")
data2=data.copy()

data2=data2.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

data2=data2.rename(columns={"v1":"Title" , "v2":"Sms"})
print(data2.groupby("Title").count())

print(data2.describe())

data2=data2.drop_duplicates()

print(data2.describe())

print(data2.isnull().sum())

data2["Number of Character"]=data2["Sms"].apply(len)
print(data2)

data2.hist(column="Number of Character", by="Title",bins=50)
plt.show()

data2.Title=[1 if code=="spam" else 0 for code in data2.Title]
print(data2)

import re

message=re.sub("[^a-zA-Z]"," ",data2["Sms"][0])
print(data2["Sms"][0])
print(message)

def letters(sentence):
    side=re.compile("[^a-zA-Z]")
    return re.sub(side," ",sentence)

print(letters("tuy,,?Rtf"))

print(data2["Sms"][0])
print(letters(data2["Sms"][0]))

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

stopping=stopwords.words("english")
print(stopping)

spam=[]
ham=[]
allsentences=[]

for i in range(len(data2["Sms"].values)):
    r1=data2["Sms"].values[i]
    r2=data2["Title"].values[i]

    cleansentences=[]
    sentences=letters(r1)
    sentences=sentences.lower()

    for words in sentences.split():
        cleansentences.append(words)
        if r2==1:
            spam.append(sentences)
        else:
            ham.append(sentences)
    
    allsentences.append(" ".join(cleansentences))

data2["New Sms"]=allsentences
print(data2)

data2=data2.drop(columns=["Sms","Number of Character"],axis=1)
print(data2)



######################### Count Vectorize, Train-Test Split, Model Identified and Optimized with Alpha #########################
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(data2["New Sms"]).toarray()

y=data2["Title"]
X=x

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(X_train,y_train)
prediction=model.predict(X_test)

from sklearn.metrics import accuracy_score
acs=accuracy_score(y_test,prediction)
print(acs*100)

for i in np.arange(0.0,1.1,0.1):
    model=MultinomialNB(alpha=i)
    model.fit(X_train,y_train)
    prediction2=model.predict(X_test)
    score=accuracy_score(y_test,prediction2)
    print("Alpha= {} ----->  Score= {}".format(round(i,1), round(score*100,2)))

######################### Count Vectorize, Train-Test Split, Model Identified and Optimized with Alpha #########################
