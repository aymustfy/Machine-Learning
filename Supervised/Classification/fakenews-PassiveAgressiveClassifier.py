import pandas as pd

data=pd.read_csv("C:/Users/MSTF/Desktop/news.csv")
data2=data.copy()

data2=data2.drop(columns="Unnamed: 0")

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(data2["text"],data2["label"],random_state=42,test_size=0.2)

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.5)


tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)

pac=PassiveAggressiveClassifier(max_iter=50,random_state=42)
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

print(confusion_matrix(y_test,y_pred))
