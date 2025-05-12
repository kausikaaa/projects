import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
diabetes_df=pd.read_csv('data/diabetes.csv')
print(diabetes_df.head())

