#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class for replies using the existing bot.

Use: Create an instance of the class and try out a text or a text and the preceding text

from keras_bot.KerasBot import KerasBot
bot = KerasBot()
bot.answer_to_text('is it going to rain?')
bot.answer_to_text('is it going to rain?', 'The sky is cloudy.')

bot.answer_to_text('Where are you from?', 'Nice to meet you')
bot.answer_to_text('Where are you from?', 'Nice to meet you')
"""
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Dropout, merge
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Activation, Dense
from keras.layers import concatenate
import numpy as np
import _pickle as cPickle
import os.path
import nltk

# Extra needed:
nltk.download('punkt')
np.random.seed(1234)  # for reproducibility


class KerasBot:

    def __init__(self):

        self._word_embedding_size = 100
        self._sentence_embedding_size = 300
        self._dictionary_size = 7000
        self._maxlen_input = 50
        self._learning_rate = 0.000001

        self._vocabulary_file = './keras_bot/vocabulary_movie'
        self._weights_file = './keras_bot/my_model_weights20.h5'
        self._weights_file_GAN = './keras_bot/my_model_weights.h5'
        self._weights_file_discrim = './keras_bot/my_model_weights_discriminator.h5'
        self._unknown_token = 'something'
        self._name_of_computer = 'john'

        self.model = self.start_model()
        self.vocabulary = cPickle.load(open(self._vocabulary_file, 'rb'))
        self.model_discr = self.init_model()


        # # Find indexes of BOS and EOS
        # self.index_EOS = [self.vocabulary.index(item) for item in self.vocabulary if item[0] == 'EOS'][0]
        # self.index_BOS = [self.vocabulary.index(item) for item in self.vocabulary if item[0] == 'BOS'][0]

    def start_model(self):

        print('Starting the model...')

        # *******************************************************************
        # Keras model of the chatbot:
        # *******************************************************************
        ad = Adam(lr=self._learning_rate)

        input_context = Input(shape=(self._maxlen_input,), dtype='int32')  # the context text'

        input_answer = Input(shape=(self._maxlen_input,), dtype='int32')  # the answer text up to the current token'

        LSTM_encoder = LSTM(self._sentence_embedding_size, kernel_initializer='lecun_uniform')  # 'Encode context'
        LSTM_decoder = LSTM(self._sentence_embedding_size,
                            kernel_initializer='lecun_uniform')  # 'Encode answer up to the current token'

        Shared_Embedding = Embedding(output_dim=self._word_embedding_size, input_dim=self._dictionary_size,
                                     input_length=self._maxlen_input)  # 'Shared'

        word_embedding_context = Shared_Embedding(input_context)
        context_embedding = LSTM_encoder(word_embedding_context)

        word_embedding_answer = Shared_Embedding(input_answer)
        answer_embedding = LSTM_decoder(word_embedding_answer)

        merge_layer = concatenate([context_embedding, answer_embedding], axis=1, )
        # 'concatenate the embeddings of the context and the answer up to current token'

        out = Dense(int(self._dictionary_size / 2), activation="relu")(merge_layer)  # 'relu activation'
        out = Dense(self._dictionary_size, activation="softmax")(
            out)  # 'likelihood of the current token using softmax activation'

        model = Model(inputs=[input_context, input_answer], outputs=[out])

        model.compile(loss='categorical_crossentropy', optimizer=ad)

        return model


    #TODO change indexes with variables (for own dictionary)
    def greedy_decoder(self, input):
        flag = 0
        prob = 1
        ans_partial = np.zeros((1, self._maxlen_input))
        ans_partial[0, -1] = 2  # the index of the symbol BOS (begin of sentence)
        for k in range(self._maxlen_input - 1):
            ye = self.model.predict([input, ans_partial]) #
            yel = ye[0, :]
            p = np.max(yel)
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
            if mp == 3:  # the index of the symbol EOS (end of sentence)
                flag = 1
            if flag == 0:
                prob = prob * p
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (self._dictionary_size - 2):
                w = self.vocabulary[k]
                text = text + w[0] + ' '
        return (text, prob)


    def preprocess(self, raw_word, name):
        l1 = ['won’t', 'won\'t', 'wouldn’t', 'wouldn\'t', '’m', '’re', '’ve', '’ll', '’s', '’d', 'n’t', '\'m', '\'re',
              '\'ve', '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .',
              '. ,', 'EOS', 'BOS', 'eos', 'bos']
        l2 = ['will not', 'will not', 'would not', 'would not', ' am', ' are', ' have', ' will', ' is', ' had', ' not',
              ' am', ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :',
              '? ', '.', ',', '', '', '', '']
        l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']
        l4 = ['jeffrey', 'fred', 'benjamin', 'paula', 'walter', 'rachel', 'andy', 'helen', 'harrington', 'kathy', 'ronnie',
              'carl', 'annie', 'cole', 'ike', 'milo', 'cole', 'rick', 'johnny', 'loretta', 'cornelius', 'claire', 'romeo',
              'casey', 'johnson', 'rudy', 'stanzi', 'cosgrove', 'wolfi', 'kevin', 'paulie', 'cindy', 'paulie', 'enzo',
              'mikey', 'i\97', 'davis', 'jeffrey', 'norman', 'johnson', 'dolores', 'tom', 'brian', 'bruce', 'john',
              'laurie', 'stella', 'dignan', 'elaine', 'jack', 'christ', 'george', 'frank', 'mary', 'amon', 'david', 'tom',
              'joe', 'paul', 'sam', 'charlie', 'bob', 'marry', 'walter', 'james', 'jimmy', 'michael', 'rose', 'jim',
              'peter', 'nick', 'eddie', 'johnny', 'jake', 'ted', 'mike', 'billy', 'louis', 'ed', 'jerry', 'alex', 'charles',
              'tommy', 'bobby', 'betty', 'sid', 'dave', 'jeffrey', 'jeff', 'marty', 'richard', 'otis', 'gale', 'fred',
              'bill', 'jones', 'smith', 'mickey']

        raw_word = raw_word.lower()
        raw_word = raw_word.replace(', ' + self._name_of_computer, '')
        raw_word = raw_word.replace(self._name_of_computer + ' ,', '')

        # replace 'can't' with 'can not' etc
        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term, l2[j])

        for term in l3:
            raw_word = raw_word.replace(term, ' ')

        for term in l4:
            raw_word = raw_word.replace(', ' + term, ', ' + name)
            raw_word = raw_word.replace(' ' + term + ' ,', ' ' + name + ' ,')
            raw_word = raw_word.replace('i am ' + term, 'i am ' + self._name_of_computer)
            raw_word = raw_word.replace('my name is' + term, 'my name is ' + self._name_of_computer)

        for j in range(30):
            raw_word = raw_word.replace('. .', '')
            raw_word = raw_word.replace('.  .', '')
            raw_word = raw_word.replace('..', '')

        for j in range(5):
            raw_word = raw_word.replace('  ', ' ')

        if raw_word[-1] != '!' and raw_word[-1] != '?' and raw_word[-1] != '.' and raw_word[-2:] != '! ' \
                and raw_word[-2:] != '? ' and raw_word[-2:] != '. ':
            raw_word = raw_word + ' .'

        if raw_word == ' !' or raw_word == ' ?' or raw_word == ' .' or raw_word == ' ! ' or raw_word == ' ? ' or raw_word == ' . ':
            raw_word = 'what ?'

        if raw_word == '  .' or raw_word == ' .' or raw_word == '  . ':
            raw_word = 'i do not want to talk about it .'

        return raw_word

    def tokenize(self, sentences):
        # Tokenizing the sentences into words:
        # tokenized_sentences = nltk.word_tokenize(sentences.decode('utf-8'))
        tokenized_sentences = nltk.word_tokenize(sentences)  # changed for error...

        index_to_word = [x[0] for x in self.vocabulary]
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
        tokenized_sentences = [w if w in word_to_index else self._unknown_token for w in tokenized_sentences]
        X = np.asarray([word_to_index[w] for w in tokenized_sentences])
        s = X.size
        Q = np.zeros((1, self._maxlen_input))
        if s < (self._maxlen_input + 1):
            Q[0, - s:] = X
        else:
            Q[0, :] = X[- self._maxlen_input:]

        return Q

    def init_model(self):
        # *******************************************************************
        # Keras model of the discriminator:
        # *******************************************************************

        ad = Adam(lr=self._learning_rate)

        input_context = Input(shape=(self._maxlen_input,), dtype='int32')  # input context
        input_answer = Input(shape=(self._maxlen_input,), dtype='int32')  # input answer
        input_current_token = Input(shape=(self._dictionary_size,))  # input_current_token

        LSTM_encoder_discriminator = LSTM(self._sentence_embedding_size,
                                          kernel_initializer='lecun_uniform')  # encoder discriminator
        LSTM_decoder_discriminator = LSTM(self._sentence_embedding_size,
                                          kernel_initializer='lecun_uniform')  # decoder discriminator

        Shared_Embedding = Embedding(output_dim=self._word_embedding_size, input_dim=self._dictionary_size,
                                     input_length=self._maxlen_input,
                                     trainable=False)  # shared

        word_embedding_context = Shared_Embedding(input_context)
        word_embedding_answer = Shared_Embedding(input_answer)
        context_embedding_discriminator = LSTM_encoder_discriminator(word_embedding_context)
        answer_embedding_discriminator = LSTM_decoder_discriminator(word_embedding_answer)
        loss = concatenate([context_embedding_discriminator, answer_embedding_discriminator, input_current_token],
                           axis=1)  # concatenation discriminator

        loss = Dense(1, activation="sigmoid")(loss)  # discriminator output

        model_discrim = Model(inputs=[input_context, input_answer, input_current_token], outputs=[loss])

        model_discrim.compile(loss='binary_crossentropy', optimizer=ad)

        if os.path.isfile(self._weights_file_discrim):
            model_discrim.load_weights(self._weights_file_discrim)

        return model_discrim

    # TODO variables for EOS and BOS index
    def run_discriminator(self, q, a):
        sa = (a != 0).sum()

        # *************************************************************************
        # running discriminator:
        # *************************************************************************

        p = 1
        m = 0
        model_discrim = self.model_discr
        count = 0

        for i, sent in enumerate(a):
            l = np.where(sent == 3)  # the position od the symbol EOS #todo was 3, made 1
            limit = l[0][0]
            count += limit + 1

        Q = np.zeros((count, self._maxlen_input))
        A = np.zeros((count, self._maxlen_input))
        Y = np.zeros((count, self._dictionary_size))

        # Loop over the training examples:
        count = 0
        for i, sent in enumerate(a):
            ans_partial = np.zeros((1, self._maxlen_input))

            # Loop over the positions of the current target output (the current output sequence):
            l = np.where(sent == 3)  # the position of the symbol EOS
            limit = l[0][0]

            for k in range(1, limit + 1):
                # Mapping the target output (the next output word) for one-hot codding:
                y = np.zeros((1, self._dictionary_size))
                y[0, int(sent[k])] = 1

                # preparing the partial answer to input:
                ans_partial[0, -k:] = sent[0:k]

                # training the model for one epoch using teacher forcing:
                Q[count, :] = q[i:i + 1]
                A[count, :] = ans_partial
                Y[count, :] = y
                count += 1

        p = model_discrim.predict([Q, A, Y])
        p = p[-sa:-1]
        P = np.sum(np.log(p)) / sa

        return P


    def answer_to_text(self, que, last_query=None):
        """Method to predict reply with the context of the previous query.
        If you add the previous query in last_query it is used for context."""

        name = '' # not sure what to do with this

        if last_query is None:
            last_query = ''
        else:
            last_query = self.preprocess(last_query, name) + ' EOS'

        que = self.preprocess(que, name)

        query = last_query + ' ' + que
        Q = self.tokenize(query)

        # Using the trained model to predict the answer:
        self.model.load_weights(self._weights_file)
        predout, prob = self.greedy_decoder(Q[0:1])
        start_index = predout.find('EOS')
        text = self.preprocess(predout[0:start_index], name) + ' EOS'

        self.model.load_weights(self._weights_file_GAN)
        predout, prob2 = self.greedy_decoder(Q[0:1])
        start_index = predout.find('EOS')
        text2 = self.preprocess(predout[0:start_index], name) + ' EOS'

        p1 = self.run_discriminator(Q, self.tokenize(text))
        p2 = self.run_discriminator(Q, self.tokenize(text2))

        if max([prob, prob2]) > .9:
            if prob > prob2:
                best = text[0:-4]
            else:
                best = text2[0:-4]
        else:
            if p1 > p2:
                best = text[0:-4]
            else:
                best = text2[0:-4]
        init = ''

        return best

