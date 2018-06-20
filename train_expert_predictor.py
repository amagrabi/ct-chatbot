"""

Train model to predict which user is an expert in an arbitrary topic.

"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from get_data import get_data

df = get_data()

# Train model to predict user
userid2name = dict(df.groupby('from.user_id')['from.name'].apply(list))
name2userid = dict(df.groupby('from.name')['from.user_id'].apply(max))

vectorizer = CountVectorizer(
    min_df=5,
    max_df=1.0,
    max_features=None,
    stop_words='english',
    lowercase=True,
    # token_pattern=r'\b\w*[a-zA-Z]{3,}\w*\b',
)
vecs = vectorizer.fit_transform(df.message.values)
user_ids = list(df['from.user_id'].values)

X_train, X_test, y_train, y_test = train_test_split(vecs, user_ids, test_size=0.2)
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


def predict_expert(topic, vectorizer=vectorizer, clf=clf, userid2name=userid2name, n=3):
    vec = vectorizer.transform([topic])
    probs = clf.predict_proba(vec)
    results = sorted(zip(clf.classes_, probs[0]), key=lambda x: x[1])[-n:]
    output = []
    for result in results:
        output.append((userid2name.get(result[0])[0], round(result[1], 3), result[0]))
    output.reverse()
    return output


def predict_topics(name, vectorizer=vectorizer, clf=clf, name2userid=name2userid, n=10):
    if name not in name2userid.keys():
        raise Exception(f'Name "{name}" does not exist in the model')
    user_id = name2userid.get(name)
    feature_names = vectorizer.get_feature_names()
    user_index = np.where(clf.classes_ == user_id)[0][0]
    results = np.argsort(clf.coef_[user_index])[-n:]
    output = []
    for result in results:
        output.append(feature_names[result])
    return output
