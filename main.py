import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
diabetes_df=pd.read_csv('data/diabetes.csv')
print(diabetes_df.head())
print(diabetes_df['Outcome'])
x=diabetes_df.drop(columns=['Outcome'],axis=1)
y=diabetes_df['Outcome']
scaler=StandardScaler()
scaler.fit(x)
std_data=scaler.transform(x)
print(std_data)
print(diabetes_df['Outcome'].value_counts())
print(diabetes_df.groupby('Outcome').mean())
# separating the data and labels
X = diabetes_df.drop(columns = 'Outcome', axis=1)
Y = diabetes_df['Outcome']
print(X)
print(Y)

#Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_df['Outcome']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)     
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')

#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

#Model Evaluation-Accuracy Score
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


#Making a Predictive System
input_data = (4,110,92,0,0,37.6,0.191,30)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')