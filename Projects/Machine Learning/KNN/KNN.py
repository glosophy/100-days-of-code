import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# read data
columns = ['age', 'blood pressure', 'specific gravity', 'albumin', 'sugar', 'red blood cells', 'pus cell',
           'pus cell clumps', 'bacteria', 'blood glucose random', 'blood urea', 'serum creatinine', 'sodium',
           'potassium', 'hemoglobin', 'packed cell volume', 'white blood cell count', 'red blood cell count',
           'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema', 'anemia', 'class']

df = pd.read_csv('chronic_kidney.csv', names=columns)

# print first rows
print(df.head())
print('-------'*10)

# check the structure of data
df.info()
print('-------'*10)

# check the null values in each column
print(df.isnull().sum())
print('-------'*10)

# encode categorical features using pandas
encode = ['specific gravity', 'albumin', 'red blood cells', 'pus cell', 'pus cell clumps', 'bacteria',
          'hypertension', 'diabetes mellitus', 'coronary artery disease', 'appetite', 'pedal edema',
          'anemia', 'class']

for i in encode:
    df[i] = pd.Categorical(df[i])
    df[i] = df[i].cat.codes

# turn rest of variables to numeric
for c in df.columns:
    if c not in encode:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# print summary of data
print(df.describe())
print('-------'*10)

# printing the dataset shape
print("Dataset No. of Rows: ", df.shape[0])
print("Dataset No. of Columns: ", df.shape[1])
print('-------'*10)

# split dataset into dependent and independent variables
y = df['class']
X = df.loc[:, df.columns != 'class']

# create a dataset

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)

# standardize numerical
