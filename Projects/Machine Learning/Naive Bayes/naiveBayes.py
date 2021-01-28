import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# read data
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
           'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
df = pd.read_csv('adult.data.csv', names=columns)
print(df.head())
print('--------------' * 5)

# print the dataset rows and columns
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('--------------' * 5)

# drop columns with '?'
df = df.replace(' ?', np.nan).dropna(axis=0, how='any')

# drop unnecessary columns
df = df.drop(columns=['fnlwgt'])

# print the dataset rows and columns
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('--------------' * 5)

# encode categorical features using pandas
encode = ['workclass', 'education', 'marital-status', 'occupation',
          'relationship', 'race', 'sex', 'native-country', 'class']
for i in encode:
    df[i] = pd.Categorical(df[i])
    df[i] = df[i].cat.codes

print(df.head(10))
print('--------------' * 5)

# print structure of the dataset
print("Dataset info:")
print(df.info())
print('--------------' * 5)

# print summary statistics
print(df.describe(include='all'))
print('--------------' * 5)

# split data into dependent and independent variables
y = df['class']
X = df.loc[:, df.columns != 'class']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define model
clf = GaussianNB()

# train
clf.fit(X_train, y_train)

# predict on test
y_pred = clf.predict(X_test)

y_pred_score = clf.predict_proba(X_test)

# evaluate
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print('--------------' * 5)

print("Accuracy : ", round(accuracy_score(y_test, y_pred) * 100, 2))
print('--------------' * 5)

print("ROC_AUC : ", round(roc_auc_score(y_test, y_pred_score[:, 1]), 2))
print('--------------' * 5)

# %%-----------------------------------------------------------------------
# confusion matrix

conf_matrix = confusion_matrix(y_test, y_pred)
class_names = df['class'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 12}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=12)
plt.title('Confusion matrix')
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)
plt.tight_layout()
plt.show()
