# XGBoost: gradient boosting on ‘steroids’

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import os

# get cwd
cwd = os.getcwd()

# read dataset
df = pd.read_csv(cwd + '/data_banknote_authentication.csv', header=None)

# assign name to columns
df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

# define target and features
X = df[['variance', 'skewness', 'curtosis', 'entropy']]
y = df['class']

# # convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
# data_dmatrix = xgb.DMatrix(data=X, label=y)

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# standarize the data: distribution will have a mean value 0 and standard deviation of 1
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# define XGBoost model
xg_reg = xgb.XGBClassifier()

xg_reg.fit(X_train_std, y_train)

y_pred = xg_reg.predict(X_test_std)


# print accuracy
print("Accuracy is ", round((accuracy_score(y_test, y_pred) * 100), 2))

# print AUC value
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
print('AUC score: ', round(roc_auc, 3))
