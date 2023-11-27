from sklearn import datasets 
from sklearn import neighbors 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import pandas as pd 
df = pd.read_csv('diabetes.csv') 
df.plot.scatter('age','bmi',c='sugar', colormap='viridis') 
X, y = df[['age','bmi']], df['sugar']
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7) 
classifier = neighbors.KNeighborsClassifier(n_neighbors=3) 
classifier.fit(x_train, y_train) 
result = classifier.predict(x_test) 
accuracy = accuracy_score(y_test, result) * 100 
print(f'Accuracy is {accuracy}') 
features = [[int(i) for i in input(f'Enter bmi, age').split()]] 
result = classifier.predict(features)
if result == 0:
  print('Not diabetic')
else:
  print('Diabetic')
