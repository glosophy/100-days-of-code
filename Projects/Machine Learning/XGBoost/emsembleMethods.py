# compare performance of different emsemble methods

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import gzip
import shutil

# get cwd
cwd = os.getcwd()

# # extract file from .gz file
# with gzip.open(cwd + '/cps_00002.dta.gz', 'rb') as f_in:
#     with open(cwd + '/cps_00002.dta', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

# read dta file
df = pd.read_stata(cwd + '/cps_00002.dta')

# print the dataset rows and columns
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('--------'*10)

# print first rows
print(df.head())
print('-------'*10)

# print structure of dataset
print(df.info())
print('--------'*10)

# turn age feature to int
df.drop(df[df['age'] == 'under 1 year'].index, inplace=True)
df.age = df.age.astype(int)

# print summary statistics
print(df.describe(include='all'))
print('--------'*10)

# look at NaNs
print("Sum of NULL values in each column:")
print(df.isnull().sum())
print('--------'*10)

# drop nan rows in dependent variable
df = df[df['incwage'].notna()]

# check number of rows left
print("New No. of Rows: ", df.shape[0])
print('--------'*10)

# look at NaNs agains
print("Sum of NULL values in each column:")
print(df.isnull().sum())
print('--------'*10)

# drop unnecessary columns
drop = ['hwtfinl', 'wtfinl', 'year', 'serial', 'month']
df.drop(drop, axis=1, inplace=True)

# check final number of rows and columns
print("Final No. of Rows: ", df.shape[0])
print("Final No. of Columns: ", df.shape[1])
print('--------'*10)

# check dependent variable distribution
plt.hist(df['incwage'])
plt.title('incwage Distribution')
plt.ylabel('Count')
plt.xlabel('incwage')
plt.show()

# identifying outliers using IQR
Q1 = df['incwage'].quantile(0.25)
Q3 = df['incwage'].quantile(0.75)
IQR = Q3 - Q1
print('Interquartile range for incwage:')
print(IQR)
print('--------'*10)

# remove outliers based on IQR
index = df[(df['incwage'] < (Q1 - 1.5 * IQR)) | (df['incwage'] > (Q3 + 1.5 * IQR))].index
df.drop(index, inplace=True)
df['incwage'].describe()

# check dependent variable
plt.hist(df['incwage'])
plt.title('incwage Distribution')
plt.ylabel('Count')
plt.xlabel('incwage')
plt.show()

# check income vs age vs sex
plt.scatter(df['age'], df['incwage'])
plt.title('incwage vs. age')
plt.ylabel('Income')
plt.xlabel('Age')
plt.show()

# check for outliers in age column
print('age column stats:')
print(df['age'].describe())
print('--------'*10)

# drop age > 80
df.drop(df[df['age'] > 80].index, inplace=True)

# check income vs age by sex
sns.scatterplot(x=df['age'], y=df['incwage'], hue=df['sex'])
plt.title('incwage vs. age, by sex')
plt.ylabel('Income')
plt.xlabel('Age')
plt.show()

# check income vs age by citizen status
sns.scatterplot(x=df['age'], y=df['incwage'], hue=df['citizen'])
plt.title('incwage vs. age, by citizen status')
plt.ylabel('Income')
plt.xlabel('Age')
plt.show()

