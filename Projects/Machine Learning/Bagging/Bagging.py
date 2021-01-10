# Bootstrap Aggregation: ensemble machine learning algorithm

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

# split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# standarize the data: distribution will have a mean value 0 and standard deviation of 1
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# initialize the base classifier: based on GridSearchCV from decision tree
base_clf = DecisionTreeClassifier(criterion='entropy',
                                  max_depth=7,
                                  min_samples_leaf=1,
                                  random_state=42)

# instantiate the bagging classifier
bgclassifier = BaggingClassifier(base_estimator=base_clf, n_estimators=100,
                                 max_samples=75,
                                 random_state=42, n_jobs=5)

# fit the bagging classifier
bgclassifier.fit(X_train_std, y_train)

# predict
y_pred = bgclassifier.predict(X_test_std)


# model scores on test and training data
# print('Model test Score: %.3f' %bgclassifier.score(X_test_std, y_test),
#       '\nModel training Score: %.3f' %bgclassifier.score(X_train_std, y_train))

# print accuracy
print("Accuracy for bagging is ", round((accuracy_score(y_test, y_pred) * 100), 2))

# print AUC value
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
print('AUC score: ', round(roc_auc, 3))
