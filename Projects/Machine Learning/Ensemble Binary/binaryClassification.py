# kaggle dataset: https://www.kaggle.com/yufengsui/portuguese-bank-marketing-data-set

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")


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

# split data into dependent and independent variables
y = df['y']
X = df.loc[:, df.columns != 'y']

# split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # parameter for gridsearchcv
# param_dict = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': range(1, 20),
#     'min_samples_split': range(2, 30),
#     'min_samples_leaf': range(1, 5)
# }
#
# grid = GridSearchCV(tree, param_grid=param_dict,
#                     cv=10, verbose=10, n_jobs=-1)
#
# grid.fit(X_train, y_train)
#
# # see best parameters
# print('Best parameters:\n', grid.best_params_)
# print('Best estimator:\n', grid.best_estimator_)
# print('Best score:\n', round(grid.best_score_, 2))

# define model
tree = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=7,
                              max_features=None, max_leaf_nodes=None,
                              min_impurity_decrease=0.0, min_impurity_split=None,
                              min_samples_leaf=2, min_samples_split=27,
                              min_weight_fraction_leaf=0.0, presort=False, random_state=42,
                              splitter='best')
bagging = BaggingClassifier(base_estimator=tree, random_state=42)
gradBoost = GradientBoostingClassifier(min_samples_leaf=2, min_samples_split=27, max_depth=7)
randomForest = RandomForestClassifier(random_state=42, max_depth=7, min_samples_leaf=2,
                                      min_samples_split=27)
XGBoost = xgb.XGBClassifier(max_depth=7, random_state=42)


models = [tree, bagging, gradBoost, randomForest, XGBoost]

cols = ['Model', 'Accuracy', 'AUC', 'Sensitivity', 'Specificity', 'Precision', 'Recall']
rows = len(models)
results = pd.DataFrame(columns=cols, index=range(rows))
model_names = ['Decision Tree', 'Bagging', 'GradBoost', 'Random Forest', 'XGBoost']
results['Model'] = model_names

k = 0
for i in models:
    # fit and predict
    i.fit(X_train, y_train)
    y_pred = i.predict(X_test)

    # evaluate
    acc = round((accuracy_score(y_test, y_pred) * 100), 2)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
    table = metrics.confusion_matrix(y_test, y_pred)
    TP = table[1, 1]
    TN = table[0, 0]
    FP = table[0, 1]
    FN = table[1, 0]
    sens = TP / float(TP + FN)
    spec = TN / float(TN + FP)
    prec = TP / float(TP + FP)
    recall = round(recall_score(y_test, y_pred, average='binary'), 2)

    # create df
    results.loc[k].Accuracy = acc
    results.loc[k].AUC = roc_auc
    results.loc[k].Sensitivity = round(sens, 2)
    results.loc[k].Specificity = round(spec, 2)
    results.loc[k].Precision = round(prec, 2)
    results.loc[k].Recall = round(recall, 2)

    k += 1

print(results)