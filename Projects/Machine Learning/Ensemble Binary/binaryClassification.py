# kaggle dataset: https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split


cwd = os.getcwd()

# read csv
df = pd.read_csv('bank-full.csv', sep=';')

# see first rows
print(df.head(10))
print('--------------' * 5)

# print df columns
print(df.columns)
print('--------------' * 5)

# print the dataset rows and columns
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('--------------' * 5)

# print structure of dataset
print(df.info())
print('--------------' * 5)

# look at NaNs
print("Sum of NULL values in each column:")
print(df.isnull().sum())
print('--------------' * 5)

# encode categorical features using pandas
encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
          'campaign', 'poutcome', 'y', 'month']

for i in encode:
    df[i] = pd.Categorical(df[i])
    df[i] = df[i].cat.codes

print(df.head(10))
print('--------------' * 5)

# print summary statistics
print(df.describe(include='all'))
print('--------------' * 5)

# check dependent variable distribution
plt.hist(df['y'])
plt.title('Dependent Variable (y) Distribution')
plt.ylabel('Count')
plt.xlabel('Dependent Variable (y)')
plt.show()

# count values: y
print('Value count for dependent variable:')
print(df['y'].value_counts())
print('--------------' * 5)

print('Percentage count for dependent variable:')
print(df['y'].value_counts(normalize=True) * 100)  # get proportion of values over total
print('--------------' * 5)

# check age
sns.boxenplot(df['age'])
plt.title('Age Boxplot')
plt.show()

# check features against dependent variable
sns.boxenplot(x='y', y='age', data=df)
plt.title('Age Boxplot by y')
plt.show()

sns.boxenplot(x='y', y='balance', data=df)
plt.title('Balance Boxplot by y')
plt.show()

sns.boxenplot(x='y', y='day', data=df)
plt.title('Day Boxplot by y')
plt.show()

sns.boxenplot(x='y', y='duration', data=df)
plt.title('Duration Boxplot by y')
plt.show()

# drop unnecessary columns: not directly related to the client
drop = ['day', 'contact', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
df.drop(drop, axis=1, inplace=True)

# split data into dependent and independent variables
y = df['y']
X = df.loc[:, df.columns != 'y']

# split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

