#Python Libraries used.
# pandas
import pandas as pd
from pandas import Series,DataFrame
from pandas import get_dummies
# tensorflow
import tensorflow as tf

# numpy, matplotlib, seaborn
import numpy as np

# machine learning
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, log_loss

# Reading the training data and test data.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# So,the columns are: id, species, margin1, margin2, margin3
# Well-conditioned data has zero mean and equal variance
data_norm = pd.DataFrame(train_df)
cols = train_df.columns

#Accuracy to measure the predictive accuracy of a model.
#Log Loss measures the performance of a classification model
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
features = cols[0:194]
labels = data_norm["species"]

#Shuffle The data
indices = data_norm.index.tolist()
indices = np.array(indices)
np.random.shuffle(indices)
X = data_norm.reindex(indices)[features]
y = labels

# One Hot Encode as a dataframe

# Generate Training and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.3)

# Convert to np arrays so that we can use with TensorFlow
X_train = np.array(X_train).astype(np.str)
X_test  = np.array(X_test).astype(np.str)
y_train = np.array(y_train).astype(np.str)
y_test  = np.array(y_test).astype(np.str)

#Check to make sure split still has 4 features and 3 labels
num_features = 195
num_labels = 990
num_hidden = 10

# Function encode with two parameters passed to it.
def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train_df, test_df)
train.head(1)

# Provides train/test indices to split data in train/test sets.
stratsplit = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

for train_index, test_index in stratsplit:
    X_train, X_test = train.values[train_index], train.values[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

# Machine Learning to find a linear combination of features.
clf = LinearDiscriminantAnalysis()

# Training part of the modeling process with LDA.
clf.fit(X_train, y_train)
name = clf.__class__.__name__
    
print("="*30)
print(name)
    
print('****Results****')
train_predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, train_predictions)			#Accuracy to measure the predictive accuracy of a model.
print("Accuracy: {:.4%}".format(accuracy))
train_predictions = clf.predict_proba(X_test)
logloss = log_loss(y_test, train_predictions)				#Log Loss measures the performance of a classification model
print("Log Loss: {}".format(logloss))
    
log_entry = pd.DataFrame([[name, accuracy*100, logloss]], columns=log_cols)
log = log.append(log_entry)