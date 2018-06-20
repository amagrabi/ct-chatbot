"""

Analyze basic stats of hipchat data.

"""

import operator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from get_data import get_data


df = get_data()

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
n = 50
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
