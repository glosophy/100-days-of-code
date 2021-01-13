from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
import os


# get cwd
cwd = os.getcwd()

# read dataset
df = pd.read_csv(cwd + '/titanic_data.csv')

# printing the dataswet rows and columns
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])

# printing the dataset obseravtions
print("Dataset first few rows:")
print(df.head(2))

# printing the struture of the dataset
print("Dataset info:")
print(df.info())

# printing the summary statistics of the dataset
print(df.describe(include='all'))

#clean the dataset
print("Sum of NULL values in each column:")
print(df.isnull().sum())

# drop unnnecessary columns
drop = ['PassengerId', 'Pclass', 'Name', 'Ticket', 'Cabin']
df.drop(drop, axis=1, inplace=True)

# encode target variable
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 0, 'Q': 2})

# drop nan rows in Age
df = df[df['Age'].notna()]
df = df[df['Embarked'].notna()]

#split the dataset
# separate the predictor and target variable
X = df.values[:, 1:]
y = df.values[:, 0]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)

#plot feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, df.iloc[:, 1:].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()

# select the training dataset on k-features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:4]]

# select the testing dataset on k-features
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:4]]

#perform training with random forest with k columns
# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)

#make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# calculate metrics with all features
print("\n")
print("Results Using All Features:")

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\n")

print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
print("ROC_AUC: ", roc_auc_score(y_test, y_pred_score[:,1]) * 100)

# calculate metrics using k features
print("\n")
print("Results Using K features:")
print("Classification Report: ")
print(classification_report(y_test, y_pred_k_features))
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)