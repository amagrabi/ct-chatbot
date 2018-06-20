"""

Train markov model to generate characteristic messages.

"""

import pandas as pd
import markovify

from get_data import get_data

df = get_data()
markov_model = markovify.Text(df.message.values)


def generate_messages(n=5):
    """Train markov model to generate characteristic messages."""
    messages = []
    for i in range(n):
        messages.append(markov_model.make_sentence())
    return messages
