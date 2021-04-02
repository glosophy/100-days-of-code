import os
import tarfile
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from pathlib import Path
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

cwd = os.getcwd()

extracted_to_path = Path.cwd()
with tarfile.open('marketing1.tar.gz') as tar:
    tar.extractall(path=cwd)

# read txt file with pandas
train = pd.read_csv(cwd + '/marketing1/train_data.txt', delimiter=',')
test = pd.read_csv(cwd + '/marketing1/train_data.txt', delimiter=',')
print(train.columns)
print('------------' * 5)

# print the dataset rows and columns
print("TRAIN - Dataset No. of Rows: ", train.shape[0])
print("TRAIN - Dataset No. of Columns: ", train.shape[1])
print('--------'*10)
print("TEST - Dataset No. of Rows: ", test.shape[0])
print("TEST - Dataset No. of Columns: ", test.shape[1])
print('--------'*10)

# look at NaNs
print("Sum of NULL values in each column (train):")
print(train.isnull().sum())
print('--------'*10)
print("Sum of NULL values in each column (test):")
print(test.isnull().sum())
print('--------'*10)

# drop 'time-in-bay-area'
train.drop('time-in-bay-area', axis=1, inplace=True)
test.drop('time-in-bay-area', axis=1, inplace=True)

# drop nan rows in dependent variable
fill_nan = ['marital-status', 'education', 'occupation', 'persons-in-household', 'householder-status', 'type-of-home']

for i in fill_nan:
    train = train[train[i].notna()]
    test = test[test[i].notna()]

# print the dataset rows and columns
print("TRAIN - Dataset No. of Rows: ", train.shape[0])
print("TRAIN - Dataset No. of Columns: ", train.shape[1])
print('--------'*10)
print("TEST - Dataset No. of Rows: ", test.shape[0])
print("TEST - Dataset No. of Columns: ", test.shape[1])
print('--------'*10)

# do some EDA on train set
train['income'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.ylabel('Count')
plt.show()

# label distribution
print('Label Distribution in %')
count_values = train['income'].value_counts().tolist()
labels_count = train['income'].value_counts().index.tolist()
for i in range(len(count_values)):
    print('Label {}:'.format(labels_count[i]), round(count_values[i] * 100 / len(train), 2))
print('------------' * 5)

# plot education by income
train.groupby('income').education.value_counts().unstack(0).plot.barh()
plt.title('Type of Education by Income')
plt.show()

# plot occupation by income
train.groupby('income').occupation.value_counts().unstack(0).plot.barh()
plt.title('Type of Ocuppation by Income')
plt.show()

# divide into train/test
y_train, x_train, y_test, x_test = train['income'], train.loc[:, train.columns != 'income'], test['income'], \
                                   test.loc[:, test.columns != 'income']

# define XGBoost model
xg_reg = xgb.XGBClassifier()

xg_reg.fit(x_train, y_train)

y_pred = xg_reg.predict(x_test)

# print accuracy
print("Accuracy is ", round((accuracy_score(y_test, y_pred) * 100), 2))
