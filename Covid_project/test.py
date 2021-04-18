import nltk
from nltk.corpus import stopwords
import re
import string

#opening a file
import pandas as pd
train = pd.read_csv('COVIDFakeNewsData.csv')
train.head()

train=train[['headlines','outcome']]
#print(train)

X_train=train['headlines']
Y_train=train['outcome']


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(X_train)
freq_term_matrix = count_vectorizer.transform(X_train)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(tf_idf_matrix,
                                   Y_train, random_state=0)

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
model=NB.fit(X_train, Y_train)
Accuracy = NB.score(X_test, Y_test)
matrix= confusion_matrix(Y_test,model.predict(X_test))
print("Accuracy of the machine: ",Accuracy)
print("matric determining TP FP TN FN",matrix)

#These some of the code are taken from the stackoverflow.