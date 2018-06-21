"""

Plugin for the Commercetools Chatbot.

"""

import pandas as pd
import pickle
import random
from will.plugin import WillPlugin
from will.decorators import respond_to, periodic, hear, randomly, route, rendered_template, require_settings

from get_data import get_data
from predict.MarkovTextGenerator import MarkovTextGenerator
from predict.ExpertPredictor import ExpertPredictor

# data_full = get_data()
# data_undersampled_recent = get_data(max_msgs_per_user=10000, undersampling_method='recent')
# data_undersampled_random = get_data(max_msgs_per_user=10000, undersampling_method='random')
data_dummy = pd.DataFrame()

# Loading data and using MarkovTextGenerator
markov_model = pickle.load(open('models/markov_model.pkl', 'rb'))
markov_text_generator = MarkovTextGenerator(data=data_dummy, model=markov_model)

# Loading data and using ExpertPredictor
model_expert_predictor = pickle.load(open('models/model_expert_predictor.pkl', 'rb'))
vectorizer_expert_predictor = pickle.load(open('models/vectorizer_expert_predictor.pkl', 'rb'))
name2userid = pickle.load(open('models/name2userid.pkl', 'rb'))
userid2name = pickle.load(open('models/userid2name.pkl', 'rb'))
expert_predictor = ExpertPredictor(data=data_dummy,
                                   model=model_expert_predictor,
                                   vectorizer=vectorizer_expert_predictor,
                                   name2userid=name2userid,
                                   userid2name=userid2name)


class CTBotPlugin(WillPlugin):

    @respond_to(r'^(help)$')
    def show_help(self, message):
        answer = """/code
    Available commands:
    "expert <topics>" - Suggests people who are experts in the respective topics (e.g. "expert scala").
    "random" - Generates a random commercetools sentence.
    "decide <options>" - Generates a random commercetools sentence (e.g. "decide coffee tea beer").
    "topics <full user name>" - Shows the words that are most frequently used by a user (e.g. "topics Amadeus Magrabi").
    "show top users" - Shows statistics on the most active users.
    "show top emotes" - Shows statistics on the most popular emotes.
    "show top messages" - Shows statistics on the most used messages.
            """
        self.reply(answer, color='gray')

    @respond_to(r'^(show top messages)$')
    def show_top_messages(self, message):
        answer = """/code
           message  messageCount
1       (thumbsup)          6604
2               ok          4195
3              yes          2306
4               :D          2197
5               :)          1540
6               +1          1508
7           thanks          1273
8                ?          1245
9              yep          1159
10              ;)          1086
        """
        self.reply(answer, color='gray')

    @respond_to(r'^(show top users)$')
    def show_top_users(self, message):
        answer = """/code
                      user  messageCount
1             Hajo Eichler         59491
2              Sven Müller         36852
3              Simon White         26111
4               Yann Simon         25709
5           Konrad Fischer         23595
6          Nicola Molinari         22996
7   Anas Ait Said Oubrahim         18592
8           Arnaud Gourlay         16329
9     Christoph Neijenhuis         12999
10         Martin Möllmann         12223
        """
        self.reply(answer, color='gray')

    @respond_to(r'^(show top emotes)$')
    def show_top_emotes(self, message):
        answer = """/code
                      user  messageCount
1             Hajo Eichler         59491
2              Sven Müller         36852
3              Simon White         26111
4               Yann Simon         25709
5           Konrad Fischer         23595
6          Nicola Molinari         22996
7   Anas Ait Said Oubrahim         18592
8           Arnaud Gourlay         16329
9     Christoph Neijenhuis         12999
10         Martin Möllmann         12223
        """
        self.reply(answer, color='gray')

    @respond_to(r'^(expert) (?P<topic>.*)')
    def find_experts(self, message, topic):
        answer = expert_predictor.predict_experts_str(topic=topic, n=3, replace_at=True)
        self.reply(answer, color='gray')

    @respond_to(r'^(topics) (?P<name>.*)')
    def find_topics(self, message, name):
        answer = expert_predictor.predict_topics_str(name, n=5, replace_at=True)
        self.reply(answer, color='gray')

    @respond_to(r'^(random)$')
    def generate_markov_text(self, message):
        answer = markov_text_generator.generate_texts_str(n=1, replace_at=True)
        self.reply(answer, color='gray')

    @respond_to(r'^(decide) (?P<options>.*)')
    def decide(self, message, options):
        options = str(options)
        options = options.split(' ')
        if len(options) >= 2:
            answer = str(random.choice(options)) + '!'
        else:
            answer = 'If I need to make a choice for you, then there have to be multiple options!'
        self.reply(answer, color='gray')
