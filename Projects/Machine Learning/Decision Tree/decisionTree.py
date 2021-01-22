# Decision Tree with banknote authentication dataset: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

# import packages
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from pydotplus import graph_from_dot_data
import webbrowser
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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

# run decision tree algorithms: gini and entropy
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42,
                                  max_depth=3, min_samples_leaf=5, class_weight='balanced')
clf_gini.fit(X_train_std, y_train)

clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42,
                                     max_depth=3, min_samples_leaf=5, class_weight='balanced')
clf_entropy.fit(X_train_std, y_train)

# prediction
y_pred = clf_gini.predict(X_test_std)
y_pred_en = clf_entropy.predict(X_test_std)

# print accuracies
print("Accuracy for Gini tree is ", round((accuracy_score(y_test, y_pred) * 100), 2))
print("Accuracy for Entropy tree is ", round((accuracy_score(y_test, y_pred_en) * 100), 2))

# classification metrics
print("\n")
print("Results using Gini Index:")
print("Classification Report: ")
print(classification_report(y_test, y_pred))
print("\n")

print("Results using Entropy:")
print("Classification Report: ")
print(classification_report(y_test, y_pred_en))
print("\n")

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = df['class'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

plt.figure(figsize=(3, 3))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 14}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.title('Confusion Matrix Gini Model')
plt.tight_layout()
plt.show()

# confusion matrix for entropy model
conf_matrix_en = confusion_matrix(y_test, y_pred_en)
df_cm_en = pd.DataFrame(conf_matrix_en, index=class_names, columns=class_names)

plt.figure(figsize=(3, 3))
hm = sns.heatmap(df_cm_en, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 14},
                 yticklabels=df_cm_en.columns, xticklabels=df_cm_en.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.title('Confusion Matrix Entropy Model')
plt.tight_layout()
plt.show()

# graph decision tree
cols = X.columns
tree_gini = export_graphviz(clf_gini, filled=True, rounded=True, feature_names=list(cols), out_file=None)
tree_entropy = export_graphviz(clf_gini, filled=True, rounded=True, feature_names=list(cols), out_file=None)

# export graph
graph_gini = graph_from_dot_data(tree_gini)
graph_entropy = graph_from_dot_data(tree_entropy)

graph_gini.write_pdf("TreeGini.pdf")
graph_entropy.write_pdf("TreeEntropy.pdf")

webbrowser.open_new(r'TreeGini.pdf')
webbrowser.open_new(r'TreeEntropy.pdf')

# get more insights by playing around with DecisionTreeClassifier( ) arguments
max_depth = [3, 6, 9, 12, 15]
min_samples_leaf = [5, 10, 15, 20, 25]

cols = ['max_depth', 'min_samples_leaf', 'acc_gini', 'acc_entropy', 'auc_gini', 'auc_entropy']
rows = len(max_depth) * len(min_samples_leaf)
df2 = pd.DataFrame(columns=cols, index=range(rows))

k = 0  # initialize counter
for i in range(len(max_depth)):
    for j in range(len(min_samples_leaf)):
        # define
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42,
                                          max_depth=max_depth[i], min_samples_leaf=min_samples_leaf[j],
                                          class_weight='balanced')

        clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42,
                                             max_depth=max_depth[i], min_samples_leaf=min_samples_leaf[j],
                                             class_weight='balanced')

        # fit
        clf_gini.fit(X_train_std, y_train)
        clf_entropy.fit(X_train, y_train)

        # predict
        y_pred = clf_gini.predict(X_test)
        y_pred_en = clf_entropy.predict(X_test)

        # evaluate
        acc_gini = round((accuracy_score(y_test, y_pred) * 100), 2)
        acc_entropy = round((accuracy_score(y_test, y_pred_en) * 100), 2)

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        false_positive_rate_en, true_positive_rate_en, thresholds_en = roc_curve(y_test, y_pred_en)

        roc_auc = round(auc(false_positive_rate, true_positive_rate), 3)
        roc_auc_en = round(auc(false_positive_rate_en, true_positive_rate_en), 3)

        # add results to df2
        df2.loc[k].max_depth = max_depth[i]
        df2.loc[k].min_samples_leaf = min_samples_leaf[j]
        df2.loc[k].acc_gini = acc_gini
        df2.loc[k].acc_entropy = acc_entropy
        df2.loc[k].auc_gini = roc_auc
        df2.loc[k].auc_entropy = roc_auc_en
        k += 1

print(df2)