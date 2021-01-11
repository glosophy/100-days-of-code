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
df = pd.read_csv(cwd + '/data_banknote_authentication.csv', header=None)

# assign name to columns
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

# define target and features
X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['class']

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# standarize the data: distribution will have a mean value 0 and standard deviation of 1
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# do training
clf.fit(X_train_std, y_train)

# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, X.columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(10, 8))
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()

# select the training dataset on k-features
newX_train = X_train_std[:, clf.feature_importances_.argsort()[::-1][:4]]

# select the testing dataset on k-features
newX_test = X_test_std[:, clf.feature_importances_.argsort()[::-1][:4]]

# specify random forest classifier
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)

# predicton on test using all features
y_pred = clf.predict(X_test_std)
y_pred_score = clf.predict_proba(X_test_std)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

# calculate metrics all features
print("\n")
print("Results Using All Features:")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# calculate metrics k features
print("\n")
print("Results Using K Features:")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)