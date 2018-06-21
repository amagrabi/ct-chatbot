"""

Train markov model to generate characteristic messages.

"""

import pandas as pd
import pickle
import os
import markovify

from get_data import get_data


class MarkovTextGenerator:

    def __init__(self, data=None, model=None):
        """Train markov model to generate characteristic messages.

        Args:
            data (pandas.DataFrame): Hipchat data.
            model (markovify.Text): Trained Markov model.

        """
        self.data = data
        self.model = model

        # Get data
        if self.data is None:
            self.data = get_data()

        # Train model
        if self.model is None:
            self.model = markovify.Text(self.data.message.values)

    def generate_texts(self, n=5):
        """Generate characteristic messages.

        Args:
            n (int): Number of texts that should be generated.

        Returns:
            list: Markov Messages.

        """
        messages = []
        for i in range(n):
            messages.append(self.model.make_sentence())
        return messages

    def generate_texts_str(self, n=5, replace_at=True):
        """Generate characteristic messages as a string.

        Args:
            n (int): Number of texts that should be generated.
            replace_at (bool): Replaces the at symbol to not accidentally ping people.

        Returns:
            str: Markov messages.

        """
        output = self.generate_texts(n=n)
        output = '\n'.join(output)
        if replace_at:
            output = output.replace('@', '(at)')
        return output

    def save(self, save_dir='models'):
        """Saves objects that can be used to load a pre-trained ExpertPredictor.

        Args:
            save_dir (str): Directory to save markov model to.

        """
        with open(os.path.join(save_dir, 'markov_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
