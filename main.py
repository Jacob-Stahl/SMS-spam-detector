import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix

df = pd.DataFrame(pd.read_csv("prep_data.csv"))

# balance, over sample fraudulent data
df_real_0 = df[df["fraudulent"] == 0]
df_fake_1 = df[df["fraudulent"] == 1]
df_fake_1_over = df_fake_1.sample(len(df_real_0), replace=True)
df = pd.concat([df_real_0, df_fake_1_over], axis = 0)

# encode text
text_enc = CountVectorizer()
df["raw_text"].apply(lambda x: x.lower)
X = text_enc.fit_transform(df["raw_text"])
Y = df["fraudulent"]

# cross validation
model = MultinomialNB()
cv_results = cross_validate(model, X, Y, cv=10)
print("cross validation average score      : ", round(cv_results['test_score'].mean(), 4))
print("cross validation standard_deviation : ", round(cv_results['test_score'].std() , 4))

# training
print("training...")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20) # 80 / 20 split

dummy = DummyClassifier(strategy="most_frequent") # dummy model for baseline performance, picks most common target
dummy.fit(X_train, Y_train)
print("baseline performance                : ", round(dummy.score(X_test, Y_test), 4))

targets = Y_train.values
model.fit(X_train, targets)

predictions = model.predict(X_test)
print("confusion matrix (0 = real, fraud = 1): ")
c_mat = confusion_matrix(Y_test.values, predictions)
print(c_mat)
print("metrics : ")
print("   false positive rate: ", round(c_mat[1, 0] / (np.sum(c_mat[1, :])), 4))
print("   false negative rate: ", round(c_mat[0, 1] / (np.sum(c_mat[0, :])), 4))