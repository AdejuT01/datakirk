import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import pickle

# load the csv file
df_train = pd.read_csv('wine_train.csv')
df_test = pd.read_csv('wine_test.csv')
print(df_train.head())

# Splitting the train features and target variable
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:, -1]

#Splitting the test features and target variable
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:, -1]

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the data in the 'response' column
X_train['type'] = label_encoder.fit_transform(X_train['type'])
X_test['type'] = label_encoder.fit_transform(X_test['type'])

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

reg = LinearRegression()
reg.fit(X_train, y_train)
reg.predict(X_test)

pickle.dump(reg, open("model.pkl", "wb"))

