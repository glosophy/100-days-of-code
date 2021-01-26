import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)

# read data
columns = ['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell',
           'pus cell clumps', 'bacteria', 'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
           'potassium', 'hemoglobin', 'packed cell volume', 'white blood cell count', 'red blood cell count',
           'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema', 'anemia', 'class']

df = pd.read_csv('chronic_kidney.csv', names=columns)

# print first rows
print(df.head())
print('-------' * 10)

# check the structure of data
df.info()
print('-------' * 10)

# check the null values in each column
print(df.isnull().sum())
print('-------' * 10)

# encode categorical features using pandas
encode = ['specific gravity', 'albumin', 'red blood cells', 'pus cell', 'pus cell clumps', 'bacteria',
          'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema',
          'anemia', 'class']

for i in encode:
    df[i] = pd.Categorical(df[i])
    df[i] = df[i].cat.codes

# turn rest of variables to numeric
for c in df.columns:
    if c not in encode:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c].fillna(df[c].mode()[0], inplace=True)

# print summary of data
print(df.describe(include='all'))
print('-------' * 10)

# printing the dataset shape
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('-------' * 10)

# count values: y
print('Value count for dependent variable:')
print(df['class'].value_counts())
print('-------' * 10)

print('Percentage count for dependent variable:')
print(df['class'].value_counts(normalize=True) * 100)  # get proportion of values over total
print('-------' * 10)

# split dataset into dependent and independent variables
y = df['class']
X = df.loc[:, df.columns != 'class']

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# define model
clf = KNeighborsClassifier(n_neighbors=3)

# train model
clf.fit(X_train, y_train)

# prediction on test
y_pred = clf.predict(X_test)

# evaluate models
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print('-------' * 10)
print("Accuracy : ", round(accuracy_score(y_test, y_pred) * 100, 2))
print('-------' * 10)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = df['class'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(4, 4))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), ha='right', fontsize=10)
plt.ylabel('True label', fontsize=10)
plt.xlabel('Predicted label', fontsize=10)
plt.title('Confusion matrix without feature normalization', fontsize=11)
plt.tight_layout()
plt.show()

# ======================================================================
# feature normalization (MinMaxScaler)
# ======================================================================
# get dummies
dummies = ['specific gravity', 'albumin', 'red blood cells', 'pus cell', 'pus cell clumps', 'bacteria',
           'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema',
           'anemia']
df_dummies = pd.get_dummies(columns=dummies, data=X)
df_normal = pd.concat([X, df_dummies], axis=1)
df_normal.drop(dummies, inplace=True, axis=1)

# normalizing features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df_normal)

# encode dependent variable
class_le = LabelEncoder()
y_scaled = class_le.fit_transform(y)

# split the dataset into train and test
X_train_sc, X_test_sc, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42, stratify=y)

# train model
clf.fit(X_train_sc, y_train)

# prediction on test
y_pred_sc = clf.predict(X_test_sc)

# evaluate models
print("Classification Report: ")
print(classification_report(y_test, y_pred_sc))
print('-------' * 10)
print("Accuracy : ", round(accuracy_score(y_test, y_pred_sc) * 100, 2))
print('-------' * 10)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_sc)
class_names = df['class'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(4, 4))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 10}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), ha='right', fontsize=10)
plt.ylabel('True label', fontsize=10)
plt.xlabel('Predicted label', fontsize=10)
plt.title('Confusion matrix with feature normalization', fontsize=11)
plt.tight_layout()
plt.show()
