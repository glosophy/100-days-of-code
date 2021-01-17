# compare performance of different emsemble methods
from sklearn import metrics
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from pydotplus import graph_from_dot_data
import xgboost as xgb
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
drop = ['hwtfinl', 'wtfinl', 'year', 'serial', 'month', 'cpsid', 'cpsidp', 'asecflag']
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

# %%-----------------------------------------------------------------------

# split data into dependent variables and features
y = df['incwage']
X = df.loc[:, df.columns != 'incwage']

# encode categorical features: dictionary mapping
encode = ['race', 'sex', 'vetstat', 'citizen', 'occ1990', 'ind1990', 'wkstat',
          'educ', 'schlcoll', 'diffany', 'migrate1', 'nchild']

for i in encode:
    labels = X[i].astype('category').cat.categories.tolist()
    replace_map_comp = {i: {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}
    # replace values in dataframe
    X.replace(replace_map_comp, inplace=True)

# split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define models
reg_tree = DecisionTreeRegressor()
reg_bagging = BaggingRegressor(base_estimator=reg_tree)
reg_adaBoost = AdaBoostRegressor()
reg_gradBoost = GradientBoostingRegressor()
reg_randomForest = RandomForestRegressor()
reg_XGBoost = xgb.XGBRegressor()

models = [reg_tree, reg_bagging, reg_adaBoost, reg_gradBoost, reg_randomForest, reg_XGBoost]

cols = ['Model', 'R2Score', 'RMSE']
rows = len(models)
results = pd.DataFrame(columns=cols, index=range(rows))

k = 0
for i in models:
    # fit and predict
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)

    # evaluate
    r2score = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(mse)
    print(rmse)
    # false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    # roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
    # table = metrics.confusion_matrix(y_test, y_pred)
    # TP = table[1, 1]
    # TN = table[0, 0]
    # FP = table[0, 1]
    # FN = table[1, 0]
    # sens = TP / float(TP + FN)
    # spec = TN / float(TN + FP)
    # prec = TP / float(TP + FP)

    # create df
    results.loc[k].Model = '{}'.format(i)
    results.loc[k].R2Score = r2score
    results.loc[k].MSE = rmse

    k += 1

print(results)
