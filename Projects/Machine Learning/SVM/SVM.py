import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# read mushroom_data as panda dataframe
mushroom_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
                            sep=',', header=None)

# define column names
mushroom_data.columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                         'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                         'stalk-surf-abv-ring', 'stalk-surf-bel-ring', 'stalk-color-abv-ring', 'stalk-color-bel-ring',
                         'veil-type', 'veil-color', 'ring-number', 'ring-type',
                         'spore-print-color', 'population', 'habitat']

# print first few rows
mushroom_data.head()

# replace missing characters as NaN
mushroom_data.replace('?', np.NaN, inplace=True)

# check the structure of mushroom_data
mushroom_data.info()

# check the null values in each column
print(mushroom_data.isnull().sum())

# print summary of data
mushroom_data.describe(include='all')

# replace categorical mushroom_data with the most frequent value in that column
mushroom_data = mushroom_data.apply(lambda x: x.fillna(x.value_counts().index[0]))

# again check the null values in each column
mushroom_data.isnull().sum()

# encode features using get dummies
X_data = pd.get_dummies(mushroom_data.iloc[:, 1:])
X = X_data.values

# encoding the dependent variable
Y_data = mushroom_data.values[:, 0]
class_le = LabelEncoder()

# fit and transform the class
y = class_le.fit_transform(Y_data)

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# define model
clf = SVC(kernel="linear")

# fit the data
clf.fit(X_train, y_train)

# predict
y_pred = clf.predict(X_test)

# evaluate the model
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print('-------' * 7)
print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)
print('-------' * 7)


# display top 20 features (top 10 max positive and negative coefficient values)
def coef_values(coef, names):
    imp = coef
    print(imp)
    imp, names = zip(*sorted(zip(imp.ravel(), names)))
    imp_pos_10 = imp[-10:]
    names_pos_10 = names[-10:]
    imp_neg_10 = imp[:10]
    names_neg_10 = names[:10]

    imp_top_20 = imp_neg_10 + imp_pos_10
    names_top_20 = names_neg_10 + names_pos_10

    plt.barh(range(len(names_top_20)), imp_top_20, align='center')
    plt.yticks(range(len(names_top_20)), names_top_20)
    plt.show()


# get the names of columns
features_names = X_data.columns

# call the function
coef_values(clf.coef_, features_names)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = mushroom_data['class'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 12}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)

# show heat map
plt.tight_layout()
plt.show()

# plot ROC Area Under Curve
y_pred_proba = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
