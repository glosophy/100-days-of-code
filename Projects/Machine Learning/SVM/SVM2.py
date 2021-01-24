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

df = pd.read_csv('voice.csv')

# print first rows
print(df.head())
print('-------'*10)

# check the structure of data
df.info()
print('-------'*10)

# check the null values in each column
print(df.isnull().sum())
print('-------'*10)

# print summary of data
print(df.describe())
print('-------'*10)

# split dataset into dependent and independent variables
y = df['label']
X = df.loc[:, df.columns != 'label']

# encode labels
class_le = LabelEncoder()
# fit and transform the class
y = class_le.fit_transform(y)

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
features_names = X.columns

# call the function
coef_values(clf.coef_, features_names)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = df['label'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(5, 5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 12}, yticklabels=df_cm.columns,
                 xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
plt.title('Confusion Matrix')
plt.ylabel('True label', fontsize=12)
plt.xlabel('Predicted label', fontsize=12)
plt.tight_layout()
plt.show()

# plot ROC Area Under Curve
y_pred_prob = clf.decision_function(X_test)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

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

