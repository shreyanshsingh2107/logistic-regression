import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


data=pd.read_csv('train.csv')
print(data.head(10))
print("no of passengers in original deta:" ,str(len(data.PassengerId)))

sns.countplot(x="Survived",data=data)
plt.title("survived passengers ")
plt.show()

sns.countplot(x="Survived",hue="Sex",data=data)
plt.title("survived passengers according to sex")
plt.show()

sns.countplot(x="Survived",hue="Pclass",data=data)
plt.title("survived passengers according to their class")
plt.show()

data["Age"].plot.hist()
plt.title("Age of passengers")
plt.show()

data["Fare"].plot.hist()
plt.title("Fare ")
plt.show()

sns.countplot(x="SibSp",data=data)
plt.show()

print(data.isnull())
print(data.isnull().sum())

sns.boxplot(x="Pclass",y="Age",data=data)
plt.show()

data.drop("Cabin",axis=1,inplace=True)


data.dropna(inplace=True)

print(data.isnull().sum())
sex=pd.get_dummies(data['Sex'],drop_first=True)

embark=pd.get_dummies(data['Embarked'],drop_first=True)


Pcl=pd.get_dummies(data['Pclass'],drop_first=True)


data=pd.concat([data,sex,embark,Pcl],axis=1)


data.drop(['Sex','Embarked','PassengerId','Name','Ticket','Pclass'],axis=1,inplace=True)
print(data)


x=data.drop("Survived",axis=1)
y=data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
logmodel=LogisticRegression()
logmodel.fit(x_train, y_train)
predictions=logmodel.predict(x_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(accuracy_score(y_test,predictions))