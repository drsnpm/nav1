import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
data = pd.read_csv('pima-indians-diabetes.csv')
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print("Total instances: ",data.shape[0])
print("Total attributes: ",data.shape[1])
from sklearn.model_selection import train_test_split
print("The First 5 instances are: \n",data.head(5))
split_ratio=0.2
X_train, X_test, y_train, y_test = train_test_split(data,y,test_size=split_ratio)
classifier = GaussianNB()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
print("Training Data: ",X_train.shape[0],"\nTesting Data: ",X_test.shape[0])
print("Accuracy is:",accuracy_score(y_pred,y_test))