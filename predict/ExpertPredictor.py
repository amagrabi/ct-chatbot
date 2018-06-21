"""

Train model to predict which user is an expert in an arbitrary topic.

"""

import numpy as np
import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from get_data import get_data


class ExpertPredictor:

    def __init__(self, data=None, model=None, vectorizer=None, userid2name=None, name2userid=None,
                 max_msgs_per_user=10000):
        """Train markov model to generate characteristic messages.

        Args:
            data (pandas.DataFrame): Hipchat data.
            model (sklearn.naive_bayes.MultinomialNB): Trained sklearn model.
            vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): Trained sklearn text vectorizer.
            userid2name (dict): Maps Hipchat user ids to user names.
            name2userid (dict): Maps Hipchat user names to user ids.
            max_msgs_per_user (int): When data is not passed and imported, this number sets a limit on the
                                     maximum messages per user by undersampling.

        """
        self.data = data
        self.model = model
        self.vectorizer = vectorizer
        self.userid2name = userid2name
        self.name2userid = name2userid
        self.max_msgs_per_user = max_msgs_per_user

        # Get data
        if self.data is None:
            self.data= get_data(max_msgs_per_user=max_msgs_per_user)
        if self.userid2name is None:
            self.userid2name = dict(self.data.groupby('from.user_id')['from.name'].apply(list))
        if self.name2userid is None:
            self.name2userid = dict(self.data.groupby('from.name')['from.user_id'].apply(max))

        # Train model
        if self.model is None:

            # Vectorize data
            self.vectorizer = TfidfVectorizer(
                min_df=1,
                max_df=1.0,
                max_features=None,
                stop_words='english',
                lowercase=True,
                # token_pattern=r'\b\w*[a-zA-Z]{3,}\w*\b',
            )
            self._vecs = self.vectorizer.fit_transform(self.data.message.values)
            self._user_ids = list(self.data['from.user_id'].values)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(self._vecs, self._user_ids, test_size=0.2)
            self.model = MultinomialNB().fit(X_train, y_train)

            # Evaluate accuracy
            y_pred = self.model.predict(X_test)
            self._accuracy = accuracy_score(y_test, y_pred)

    def predict_experts(self, topic, n=3):
        """Predicts n experts for a specific topic.

        Args:
            topic (str): A topic or a sentence for which experts should be found.
            n (int): Number of experts that should be found.

        Returns:
            list: HipChat names of experts.

        """
        vec = self.vectorizer.transform([topic])
        probs = self.model.predict_proba(vec)
        results = sorted(zip(self.model.classes_, probs[0]), key=lambda x: x[1])[-n:]
        output = []
        for result in results:
            output.append((self.userid2name.get(result[0])[0], round(result[1], 3), result[0]))
        output.reverse()
        return output

    def predict_experts_str(self, topic, n=3, replace_at=True):
        """Predicts n experts for a specific topic and output a string.

        Args:
            topic (str): A topic or a sentence for which experts should be found.
            n (int): Number of experts that should be found.
            replace_at (bool): Replaces the at symbol to not accidentally ping people.

        Returns:
            str: HipChat names of experts.

        """
        results = self.predict_experts(topic=topic, n=n)
        output = ''
        for i, result in enumerate(results):
            output += f'{i+1}. {result[0]} \n'
        output = output.strip()
        if replace_at:
            output = output.replace('@', '(at)')
        return output

    def predict_topics(self, name, n=5):
        """Predicts n topics that are indicative of a HipChat user.

        Args:
            name (str): HipChat user for which topics should be found.
            n (int): Number of topics that should be found.

        Returns:
            list: Topics of the HipChat user.

        """
        if name not in self.name2userid.keys():
            raise Exception(f'Name "{name}" does not exist in the model')
        user_id = self.name2userid.get(name)
        feature_names = self.vectorizer.get_feature_names()
        user_index = np.where(self.model.classes_ == user_id)[0][0]
        results = np.argsort(self.model.coef_[user_index])[-n:]
        output = []
        for result in results:
            output.append(feature_names[result])
        return output

    def predict_topics_str(self, name, n=3, replace_at=True):
        """Predicts n topics that are indicative of a HipChat user and output a string.

        Args:
            name (str): HipChat user for which topics should be found.
            n (int): Number of topics that should be found.
            replace_at (bool): Replaces the at symbol to not accidentally ping people.

        Returns:
            list: Topics of the HipChat user.

        """
        results = self.predict_topics(name=name, n=n)
        output = ''
        for i, result in enumerate(results):
            output += f'{i+1}. {result} \n'
        output = output.strip()
        if replace_at:
            output = output.replace('@', '(at)')
        return output

    def save(self, save_dir='models'):
        """Saves objects that can be used to load a pre-trained ExpertPredictor.

        Args:
            save_dir (str): Directory to save files to.

        """
        with open(os.path.join(save_dir, 'model_expert_predictor.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(save_dir, 'vectorizer_expert_predictor.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(save_dir, 'userid2name.pkl'), 'wb') as f:
            pickle.dump(self.userid2name, f)
        with open(os.path.join(save_dir, 'name2userid.pkl'), 'wb') as f:
            pickle.dump(self.name2userid, f)
