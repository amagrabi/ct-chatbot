"""

Analyze basic stats of hipchat data.

"""

import markovify
import numpy as np
import operator
import pandas as pd
from pathlib import Path
import re
from sklearn.feature_extraction.text import CountVectorizer

path_data = Path.cwd() / 'data' / 'data.pkl'

df = pd.read_pickle(str(path_data))

# Drop useless columns
df.drop(columns=['data_filename', 'data_filepath', 'file.name', 'file.size', 'file.url'], inplace=True)

# See messages of specific user
# df.message[df['from.name'] == 'commercetools GmbH · Docker Monitor'].values.tolist()

# Drop automated messages
bots = ['GitHub', 'Docker Monitor', 'TeamCity', 'Jenkins', 'JIRA', 'Sphere CI', 'Frontend Bot', 'Travis CI',
        'Prometheus · AlertManager', 'Sphere Staging', 'Sphere Production', 'Subversion', 'Grafana', 'grafana',
        'Standup', 'AnomalyDetector', 'PagerDuty', 'UserVoice', 'Confluence', 'MMS',
        'commercetools GmbH · Docker Monitor', 'Frontend Production', 'Mailroom', 'Stackdriver',
        'Prometheus alerts · AlertManager', 'LaunchDarkly', 'Mailroom · system@iron.io', 'Ru Bot',
        'logentries-alerts', 'HipGif', 'commercetools GmbH · GitHub', 'Status IO', 'StatusPage.io', 'ROSHAMBO!',
        'commercetools GmbH · TeamCity', 'appear.in', 'commercetools GmbH · Travis CI', 'Integrations',
        'Sphere Frontend', 'commercetools GmbH · Datadog', 'commercetools GmbH · Jenkins', 'System',
        'commercetools GmbH · Automation', 'commercetools GmbH · Auto Mation', 'commercetools GmbH · akamel',
        'commercetools GmbH · Subversion', 'commercetools GmbH · Heroku', 'Send from CLI',
        'AgileZen', 'Log Entries', 'Link', 'Guggy', 'Automation', 'lunchbot']
df = df[~df['from.name'].isin(bots)]

# Deal with multiple and trailing whitespaces, exclude empty messages
df.message = df.message.apply(lambda x: re.sub(' +',' ', x).strip())
df = df[~df.message.isin(['', ' '])]
# TODO remove automated messages from users like Simons "@here The ES Listing Validator has news, but..."

# Number of total messages
len(df)

# Most active users
n = 20
top_users = pd.DataFrame(df['from.name'].value_counts())
top_users.reset_index(inplace=True)
top_users.rename(columns={'index': 'user', 'from.name': 'messageCount'}, inplace=True)
top_users.head(n)

# Most active rooms
n = 20
top_rooms = pd.DataFrame(df['room'].value_counts())
top_rooms.reset_index(inplace=True)
top_rooms.rename(columns={'index': 'room', 'room': 'messageCount'}, inplace=True)
top_rooms.head(n)

# Most common messages
n = 20
top_messages = pd.DataFrame(df['message'].value_counts())
top_messages.reset_index(inplace=True)
top_messages.rename(columns={'index': 'message', 'from.name': 'messageCount'}, inplace=True)
top_messages.head(n)

# Most common messages of top users
n = 10
for user in top_users.user.values[:n]:
    print(f'------- User: {user} -------')
    top_messages = pd.DataFrame(df.message[df['from.name'] == user].value_counts())
    top_messages.reset_index(inplace=True)
    top_messages.rename(columns={'index': 'message', 'message': 'messageCount'}, inplace=True)
    print(top_messages.head(20))

# Most common messages of top rooms
n = 10
for room in top_rooms.room.values[:n]:
    print(f'------- Room: {room} -------')
    top_messages = pd.DataFrame(df.message[df['room'] == room].value_counts())
    top_messages.reset_index(inplace=True)
    top_messages.rename(columns={'index': 'message', 'message': 'messageCount'}, inplace=True)
    print(top_messages.head(20))

# Most common words
# TODO improve preprocessing with spacy
n = 20
vectorizer = CountVectorizer(stop_words='english')
messages = vectorizer.fit_transform(df.message.values)
vocab = vectorizer.vocabulary_
vocab_sorted = sorted(vocab.items(), reverse=True, key=operator.itemgetter(1))
top_words = pd.DataFrame(vocab_sorted)
top_words.reset_index(inplace=True)
top_words.rename(columns={'index': 'message', 'from.name': 'messageCount'}, inplace=True)
top_words.head(n)

# Most common emoji
n = 20
top_emojis = df[df.message.str.match(r'^\(\w+\)$')]
top_emojis = pd.DataFrame(top_emojis['message'].value_counts())
top_emojis.reset_index(inplace=True)
top_emojis.rename(columns={'index': 'message', 'from.name': 'messageCount'}, inplace=True)
top_emojis.head(n)

# Markov model to generate characteristic messages
model = markovify.Text(df.message.values)
for i in range(10):
    print(model.make_sentence())
