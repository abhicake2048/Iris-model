import pandas as pd
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
 


l = load_iris()
x = l.data
y = l.target
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
sk = model.predict([l.data[0]])
print(sk)
