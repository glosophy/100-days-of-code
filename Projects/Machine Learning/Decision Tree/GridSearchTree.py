import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

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

# define model
tree = DecisionTreeClassifier()
tree.fit(X_train_std, y_train)

# parameter for gridsearchcv
param_dict = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 20),
    'min_samples_split': range(2, 30),
    'min_samples_leaf': range(1, 5)
}

grid = GridSearchCV(tree, param_grid=param_dict,
                    cv=10, verbose=10, n_jobs=-1)

grid.fit(X_train_std, y_train)

# see best parameters
print('Best parameters:\n', grid.best_params_)
print('Best estimator:\n', grid.best_estimator_)
print('Best score:\n', round(grid.best_score_, 2))