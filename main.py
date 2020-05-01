import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix

np.random.seed(1)
df = pd.DataFrame(pd.read_csv("prep_data.csv"))

# balance, under sample real data
real_fake_ratio = 1
print("balancing data...")
df_real = df[df["fraudulent"] == 0]
df_fake = df[df["fraudulent"] == 1]
df_real_under = df_real.sample(int(len(df_fake) * real_fake_ratio), replace=True)
df = pd.concat([df_fake, df_real_under], axis = 0)

# encode text
print("encoding text...")
text_enc = CountVectorizer()
df["raw_text"].apply(lambda x: x.lower)
X = text_enc.fit_transform(df["raw_text"])
Y = df["fraudulent"]

# cross validation
model = MultinomialNB()
cv_results = cross_validate(model, X, Y, cv=4)
print("cross validation using naive bayes")
print("cross validation average accuracy   : ", round(cv_results['test_score'].mean(), 4))
print("cross validation standard deviation : ", round(cv_results['test_score'].std() , 4))

# training
print("training...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .30)

# dummy
print("Dummy")
dummy = DummyClassifier(strategy="most_frequent") # dummy model for baseline performance, picks most common target in the training set
dummy.fit(X_train, Y_train)
print("baseline accuracy                   : ", round(dummy.score(X_test, Y_test), 4))
print()

# naive bayes
print("Naive Bayes")
targets = Y_train.values
model.fit(X_train, targets)
print("model accuracy                      : ", round(model.score(X_test, Y_test), 4))
predictions = model.predict(X_test)
print("confusion matrix (real = 0, fraud = 1): ")
c_mat = confusion_matrix(Y_test.values, predictions)
print(c_mat)
print("metrics : ")
print("   false positive rate: ", round(c_mat[0, 1] / (np.sum(c_mat[0, :])), 4))
print("   false negative rate: ", round(c_mat[1, 0] / (np.sum(c_mat[1, :])), 4))

# decision tree
print()
print("Decision Tree")
model = DecisionTreeClassifier()

targets = Y_train.values
model.fit(X_train, targets)
print("model accuracy                      : ", round(model.score(X_test, Y_test), 4))
predictions = model.predict(X_test)
print("confusion matrix (real = 0, fraud = 1): ")
c_mat = confusion_matrix(Y_test.values, predictions)
print(c_mat)
print("metrics : ")
print("   false positive rate: ", round(c_mat[0, 1] / (np.sum(c_mat[0, :])), 4))
print("   false negative rate: ", round(c_mat[1, 0] / (np.sum(c_mat[1, :])), 4))