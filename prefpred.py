import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

ads_data = pd.read_csv('advertising.csv')
# ads_data.head()
# ads_data.info()
# ads_data.describe()

# Create histogram of age

# ads_data['Age'].plot.hist(bins=30)

# sns.jointplot(x='Age', y='Area Income', data=ads_data)
# sns.jointplot(x='Age', y='Daily Time Spent on Site',data=ads_data,kind='kde',color='red')
# sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ads_data)
# sns.pairplot(ads_data, hue='Clicked on Ad')

# Data visualisation part ends here

# Logistic Regression part
# ads_data.head()

# Model features
X = ads_data[['Age', 'Daily Time Spent on Site', 'Area Income', 'Daily Internet Usage', 'Male']]
# Target model. Prediction
y = ads_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()

# Training model
logmodel.fit(X_test, y_train)
predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
