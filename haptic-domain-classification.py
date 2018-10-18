import collections
import operator
import seaborn as sns
import re
import matplotlib.pyplot as plt
import string
import time
import numpy as np
import pandas as pd
from nltk import word_tokenize, WordNetLemmatizer, PorterStemmer, RegexpTokenizer, TreebankWordTokenizer
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('F:/train_data.csv')
test = pd.read_csv('F:/test_data.csv')


def preprocess(df):
    p_stemmer = PorterStemmer()

    tbt = TreebankWordTokenizer()
    custom_en_stop = ['want', 'go', 'hey', 'also', 'ok']

    df = df.apply(lambda row: row.lower())
    df = df.apply(lambda row: re.sub('{.+}', '', row))
    df = df.apply(lambda row: re.sub("[0-9]{1,2} ?(am|pm)", "timeofday", row))
    df = df.apply(lambda row: re.sub("[0-9]{1,2} ?(hours?|hrs?|mins?|minutes?)", "durationtext", row))
    df = df.apply(lambda row: re.sub("[0-9]{10}\D", "phoneorpnr", row))
    df = df.apply(lambda row: word_tokenize(row))
    df = df.apply(lambda row: [WordNetLemmatizer().lemmatize(i) for i in row])
    df = df.apply(lambda row: [i for i in row if i not in string.punctuation])
    df = df.apply(lambda row: [i for i in row if i not in custom_en_stop])
    df = df.apply(lambda x: ' '.join(x))

    return df


def xy_seperator(df):
    X = df.iloc[:, 0]
    y = df.iloc[:, 1:].astype(str).replace({'T': 1, 'F': 0})
    return X, y


X_train, y_train = xy_seperator(train)
X_test, y_test = xy_seperator(test)


def plot_yhist(y):
    freq = y.sum(axis=0)
    yhist = pd.DataFrame({'label': y.columns, 'count': freq})
    # yhist['normalized'] = yhist['count'] / yhist['count'].sum()
    sns.set_style("dark")
    sns.set_palette("Reds")
    sns.barplot(x="label", y="count", data=yhist)
    plt.show()


# plot_yhist(y_train)
# plot_yhist(y_test)

LABELS = ['food', 'recharge', 'support', 'travel', 'reminders', 'nearby',
          'movies', 'casual', 'other']
dummy_labels_train = pd.get_dummies(train[LABELS])
dummy_labels_test = pd.get_dummies(test[LABELS])

X_train = preprocess(X_train)
X_test = preprocess(X_test)

tokenizer = RegexpTokenizer(r'\w+')
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=35000)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_train = preprocessing.normalize(tfidf_train)

tfidf_test = tfidf_vectorizer.transform(X_test)
tfidf_test = preprocessing.normalize(tfidf_test)

clf = OneVsRestClassifier(LogisticRegression(random_state=42, C=10))
# clf=RandomForestClassifier(random_state=42,n_estimators=200,max_depth=2750,min_samples_split=5,max_features=0.008)
clf.fit(tfidf_train, dummy_labels_train)
y_predictions = clf.predict(tfidf_test)
print('Logistic Regression Accuracy Score : ')
print(accuracy_score(y_predictions, dummy_labels_test))
print('Logistic Regression Training Accuracy Score : ')
print(clf.score(tfidf_train, dummy_labels_train))
print('Logistic Regression Classification Report : ')
print(classification_report(dummy_labels_test, y_predictions))


clf = OneVsRestClassifier(LinearSVC())
# clf=RandomForestClassifier(random_state=42,n_estimators=200,max_depth=2750,min_samples_split=5,max_features=0.008)
clf.fit(tfidf_train, dummy_labels_train)
y_predictions = clf.predict(tfidf_test)
print('LinearSVC Accuracy Score : ')
print(accuracy_score(y_predictions, dummy_labels_test))
print('LinearSVC Training Accuracy Score : ')
print(clf.score(tfidf_train, dummy_labels_train))
print('LinearSVC Classification Report : ')
print(classification_report(dummy_labels_test, y_predictions))

tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), norm='l1',
                                   max_features=25000, tokenizer=tokenizer.tokenize)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

tfidf_test = tfidf_vectorizer.transform(X_test)

y_trai = train.iloc[:, 1:].astype(str).replace({'T': 1, 'F': 0})
y_trai = y_trai.idxmax(axis=1)
y_trai = pd.Series(y_trai)
y_trai = LabelEncoder().fit_transform(y_trai)

y_tes = test.iloc[:, 1:].astype(str).replace({'T': 1, 'F': 0})
y_tes = y_tes.idxmax(axis=1)
y_tes = pd.Series(y_tes)
y_tes = LabelEncoder().fit_transform(y_tes)

alpha = [0.001, 0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
for i in alpha:
    mnb = MultinomialNB(alpha=i, fit_prior=False)
    mnb.fit(tfidf_train, y_trai)
    y_pred_class = mnb.predict(tfidf_test)
    print(accuracy_score(y_tes, y_pred_class))
