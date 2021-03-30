import os
import tarfile
import matplotlib.pyplot as plt
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

# run NN
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)

kfold = KFold(n_splits=10, shuffle=True)

results = cross_val_score(estimator, x_train, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

