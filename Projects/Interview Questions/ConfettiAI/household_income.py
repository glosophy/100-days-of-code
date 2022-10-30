import os
import tarfile
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import cohen_kappa_score, f1_score
import tensorflow as tf
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
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path=cwd)

# read txt file with pandas
train = pd.read_csv(cwd + '/marketing1/train_data.txt', delimiter=',')
test = pd.read_csv(cwd + '/marketing1/train_data.txt', delimiter=',')
print(train.columns)
print('------------' * 5)

# print the dataset rows and columns
print("TRAIN - Dataset No. of Rows: ", train.shape[0])
print("TRAIN - Dataset No. of Columns: ", train.shape[1])
print('--------' * 10)
print("TEST - Dataset No. of Rows: ", test.shape[0])
print("TEST - Dataset No. of Columns: ", test.shape[1])
print('--------' * 10)

# look at NaNs
print("Sum of NULL values in each column (train):")
print(train.isnull().sum())
print('--------' * 10)
print("Sum of NULL values in each column (test):")
print(test.isnull().sum())
print('--------' * 10)

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
print('--------' * 10)
print("TEST - Dataset No. of Rows: ", test.shape[0])
print("TEST - Dataset No. of Columns: ", test.shape[1])
print('--------' * 10)

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

# # divide into train/test
# y_train, x_train, y_test, x_test = train['income'].values, train.loc[:, train.columns != 'income'].values, \
#                                    test['income'].values, test.loc[:, test.columns != 'income'].values
#
# print(y_train.shape)
# print(x_train.shape)
# print(x_train)
#
#
# x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
# print(x_test.shape)
# print(x_train.shape)
# print(x_train)
#
# y_train, y_test = to_categorical(y_train, num_classes=9), to_categorical(y_test, num_classes=9)
#
# # y_integers = np.argmax(y_train, axis=1)
#
# # model = Sequential([
# #     Dense(500, input_dim=7079, activation="relu"),
# #     Dense(250, input_dim=7079, activation="relu"),
# #     Dense(250, input_dim=7079, activation="relu"),
# #     Dense(300, input_dim=7079, activation="relu"),
# #     Dense(100, input_dim=7079, activation="relu"),
# #     Dense(9, activation="softmax")
# # ])
# # model.compile(optimizer=Adam(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])
# #
# # model.fit(x_train, y_train, batch_size=64, epochs=20, validation_data=(x_test, y_test))
# #
# # print("Final accuracy on validations set:", 100 * model.evaluate(x_test, y_test)[1], "%")
# # print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
# # print("F1 score", f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))

train_data = train.values
test_data = test.values

x_train, y_train = train_data[:, 1:], train_data[:, 0]
x_test, y_test = test_data[:, 1:], test_data[:, 0]

print(np.unique(y_train))
print(len(np.unique(y_train)))

# https://keras.io/api/utils/python_utils/#to_categorical-function
y_train = tf.keras.utils.to_categorical(y_train-1, num_classes=9)
y_test = tf.keras.utils.to_categorical(y_test-1, num_classes=9)

print('Sets shape:')
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
print('------------' * 5)

model = Sequential([
    Dense(32, activation='relu', input_shape=(9,)),
    Dense(32, activation='relu'),
    Dense(9, activation='softmax'),
])

model.compile(optimizer=Adam(lr=0.01), loss="categorical_crossentropy", metrics=["accuracy"])

hist = model.fit(x_train, y_train,
                 batch_size=32, epochs=20,
                 validation_data=(x_test, y_test))

