# -*- coding: utf-8 -*-
"""
Training bot on own dataset.
Based on https://github.com/oswaldoludwig/Seq2seq-Chatbot-for-Keras, adaptated to pyrhton 3.6 and corrections.
author/adaptaion @evilaz

Data to be downloaded: see Readme (weights and glove)
-From https://www.dropbox.com/sh/o0rze9dulwmon8b/AAA6g6QoKM8hBEHGst6W4JGDa?dl=0
weights files: "my_model_weights20.h5", "my_model_weights.h5", "my_model_weights_discriminator.h5"
-From https://nlp.stanford.edu/projects/glove/ download the Glove folder 'glove.6B'

The authors' ready bot can run with 'python conversation_discriminator.py'
This script is for tuning own data

The instructions:
To train a new model or to fine tune on your own data:
1.If you want to train from the scratch, delete the file my_model_weights20.h5. To fine tune on your data, keep this file;
2.Download the Glove folder 'glove.6B' and include this folder in the directory of the chatbot (you can find this folder here).
This algorithm applies transfer learning by using a pre-trained word embedding, which is fine tuned during the training;
3. Run split_qa.py to split the content of your training data into two files: 'context' and 'answers' and get_train_data.py
to store the padded sentences into the files 'Padded_context' and 'Padded_answers';
4. Run train_bot.py to train the chatbot (it is recommended the use of GPU, to do so type: THEANO_FLAGS=mode=FAST_RUN,
device=gpu,floatX=float32,exception_verbosity=high python train_bot.py);
Name your training data as "data.txt". This file must contain one dialogue utterance per line.
If your dataset is big, set the variable num_subsets (in line 29 of train_bot.py) to a larger number.
weights_file = 'my_model_weights20.h5'
weights_file_GAN = 'my_model_weights.h5'
weights_file_discrim = 'my_model_weights_discriminator.h5'
"""
import numpy as np
import pandas as pd
import os
import nltk
import itertools
import operator
import pickle
from keras.layers import Input, Embedding, LSTM, Dense, RepeatVector, Bidirectional, Dropout, merge
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.models import Sequential
from keras.layers import Activation, Dense, concatenate
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
# import cPickle
import _pickle as cPickle #workaround for python3

np.random.seed(1234)  # for reproducibility


def get_hipchat_room_data_file(room='Berlin'):
    """Get all messages from a hipchat room to txt file."""

    df = pd.read_csv('data/data.csv', index_col=0)
    df_room = df[df['room'] == room]
    messages = df_room['message']

    f = open("data/messages", "w")
    mylist = list(messages)
    f.write("\n".join(map(lambda x: str(x), mylist)))
    f.close()


def split_qa(textfile='data/messages'):
    """Splits a file with text messages from conversations to context and answers text files"""

    text = open(textfile, 'r')
    q = open('./keras_bot/context', 'w')
    a = open('./keras_bot/answers', 'w')
    pre_pre_previous_raw = ''
    pre_previous_raw = ''
    previous_raw = ''
    person = ' '
    previous_person = ' '

    l1 = ['won’t','won\'t','wouldn’t','wouldn\'t','’m', '’re', '’ve', '’ll', '’s','’d', 'n’t', '\'m', '\'re', '\'ve',
          '\'ll', '\'s', '\'d', 'can\'t', 'n\'t', 'B: ', 'A: ', ',', ';', '.', '?', '!', ':', '. ?', ',   .', '. ,',
          'EOS', 'BOS', 'eos', 'bos']
    l2 = ['will not','will not','would not','would not',' am', ' are', ' have', ' will', ' is', ' had', ' not', ' am',
          ' are', ' have', ' will', ' is', ' had', 'can not', ' not', '', '', ' ,', ' ;', ' .', ' ?', ' !', ' :', '? ',
          '.', ',', '', '', '', '']
    l3 = ['-', '_', ' *', ' /', '* ', '/ ', '\"', ' \\"', '\\ ', '--', '...', '. . .']

    for i, raw_word in enumerate(text):
        pos = raw_word.find('+++$+++')

        if pos > -1:
            person = raw_word[pos+7:pos+10]
            raw_word = raw_word[pos+8:]
        while pos > -1:
            pos = raw_word.find('+++$+++')
            raw_word = raw_word[pos+2:]

        raw_word = raw_word.replace('$+++','')
        previous_person = person

        for j, term in enumerate(l1):
            raw_word = raw_word.replace(term, l2[j])

        for term in l3:
            raw_word = raw_word.replace(term, ' ')

        raw_word = raw_word.lower()

        if i>0:
            q.write(pre_previous_raw[:-1] + ' ' + previous_raw[:-1]+ '\n')  # python will convert \n to os.linese
            a.write(raw_word[:-1] + '\n')

        pre_pre_previous_raw = pre_previous_raw
        pre_previous_raw = previous_raw
        previous_raw = raw_word

    q.close()
    a.close()

    print('Splitting done.')


def get_train_data():
    """Stores the padded sentences into the files 'Padded_context' and 'Padded_answers'"""

    np.random.seed(1234)  # for reproducibility
    questions_file = './keras_bot/context'
    answers_file = './keras_bot/answers'
    vocabulary_file = './keras_bot/vocabulary_movie'
    padded_questions_file = './keras_bot/Padded_context'
    padded_answers_file = './keras_bot/Padded_answers'
    unknown_token = 'something'

    vocabulary_size = 7000
    max_features = vocabulary_size
    maxlen_input = 50
    maxlen_output = 50  # cut texts after this number of words

    print("Reading the context data...")
    q = open(questions_file, 'r')
    questions = q.read()
    print("Reading the answer data...")
    a = open(answers_file, 'r')
    answers = a.read()
    all = answers + questions
    print("Tokenazing the answers...")
    paragraphs_a = [p for p in answers.split('\n')]
    paragraphs_b = [p for p in all.split('\n')]
    paragraphs_a = ['BOS ' + p + ' EOS' for p in paragraphs_a]
    paragraphs_b = ['BOS ' + p + ' EOS' for p in paragraphs_b]
    paragraphs_b = ' '.join(paragraphs_b)
    tokenized_text = paragraphs_b.split()
    paragraphs_q = [p for p in questions.split('\n')]
    tokenized_answers = [p.split() for p in paragraphs_a]
    tokenized_questions = [p.split() for p in paragraphs_q]

    # Counting the word frequencies:
    word_freq = nltk.FreqDist(itertools.chain(tokenized_text))
    print ("Found %d unique words tokens." % len(word_freq.items()))

    # I think the correct is this: ###!!!! yes or no....
    # Getting the most common words and build index_to_word and word_to_index vectors:
    vocab = word_freq.most_common(vocabulary_size-1)
    # Saving vocabulary:
    with open(vocabulary_file, 'wb') as v:
        pickle.dump(vocab, v)

    # It was originally loading the existing file
    vocab = pickle.load(open(vocabulary_file, 'rb'))  # this was originally..

    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary of size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replacing all words not in our vocabulary with the unknown token:
    for i, sent in enumerate(tokenized_answers):
        tokenized_answers[i] = [w if w in word_to_index else unknown_token for w in sent]

    for i, sent in enumerate(tokenized_questions):
        tokenized_questions[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Creating the training data:
    X = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_questions])
    Y = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_answers])

    Q = sequence.pad_sequences(X, maxlen=maxlen_input)
    A = sequence.pad_sequences(Y, maxlen=maxlen_output, padding='post')

    with open(padded_questions_file, 'wb') as q:
        pickle.dump(Q, q)

    with open(padded_answers_file, 'wb') as a:
        pickle.dump(A, a)


def train_bot():
    """Train model with own data"""

    word_embedding_size = 100
    sentence_embedding_size = 300
    dictionary_size = 7000
    maxlen_input = 50
    maxlen_output = 50
    num_subsets = 1
    Epochs = 2 #was 100
    BatchSize = 128  # Check the capacity of your GPU
    Patience = 0
    dropout = .25
    n_test = 100

    vocabulary_file = './keras_bot/vocabulary_movie'
    questions_file = './keras_bot/Padded_context'
    answers_file = './keras_bot/Padded_answers'
    weights_file = './keras_bot/my_model_weights20.h5'
    GLOVE_DIR = './keras_bot/glove/'

    early_stopping = EarlyStopping(monitor='val_loss', patience=Patience)

    def print_result(input, index_BOS=2):

        ans_partial = np.zeros((1, maxlen_input))
        ans_partial[0, -1] = index_BOS  # the index of the symbol BOS (begin of sentence) #TODO that might have changed as well..
        for k in range(maxlen_input - 1):
            ye = model.predict([input, ans_partial])
            mp = np.argmax(ye)
            ans_partial[0, 0:-1] = ans_partial[0, 1:]
            ans_partial[0, -1] = mp
        text = ''
        for k in ans_partial[0]:
            k = k.astype(int)
            if k < (dictionary_size - 2):
                w = vocabulary[k]
                text = text + w[0] + ' '
        return (text)

    # **********************************************************************
    # Reading a pre-trained word embedding and addapting to our vocabulary:
    # **********************************************************************

    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((dictionary_size, word_embedding_size))

    # Loading our vocabulary:
    vocabulary = cPickle.load(open(vocabulary_file, 'rb'))

    # Find indexes of BOS and EOS
    index_EOS = [vocabulary.index(item) for item in vocabulary if item[0] == 'EOS'][0]
    index_BOS = [vocabulary.index(item) for item in vocabulary if item[0] == 'BOS'][0]

    # Using the Glove embedding:
    i = 0
    for word in vocabulary:
        embedding_vector = embeddings_index.get(word[0])

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        i += 1

    # *******************************************************************
    # Keras model of the chatbot:
    # *******************************************************************

    ad = Adam(lr=0.00005)

    input_context = Input(shape=(maxlen_input,), dtype='int32')  # name='input_context'
    input_answer = Input(shape=(maxlen_input,), dtype='int32')  # name='input_answer'
    LSTM_encoder = LSTM(sentence_embedding_size, init='lecun_uniform')
    LSTM_decoder = LSTM(sentence_embedding_size, init='lecun_uniform')

    if os.path.isfile(weights_file):
        Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size,
                                     input_length=maxlen_input)
    else:
        Shared_Embedding = Embedding(output_dim=word_embedding_size, input_dim=dictionary_size,
                                     weights=[embedding_matrix], input_length=maxlen_input)
    word_embedding_context = Shared_Embedding(input_context)
    context_embedding = LSTM_encoder(word_embedding_context)

    word_embedding_answer = Shared_Embedding(input_answer)
    answer_embedding = LSTM_decoder(word_embedding_answer)

    merge_layer = concatenate([context_embedding, answer_embedding], axis=-1)

    out = Dense(int(dictionary_size / 2), activation="relu")(merge_layer)
    out = Dense(dictionary_size, activation="softmax")(out)

    model = Model(input=[input_context, input_answer], output=[out])

    model.compile(loss='categorical_crossentropy', optimizer=ad)

    if os.path.isfile(weights_file):
        model.load_weights(weights_file)

    # ************************************************************************
    # Loading the data:
    # ************************************************************************
    q = cPickle.load(open(questions_file, 'rb'))
    a = cPickle.load(open(answers_file, 'rb'))
    n_exem, n_words = a.shape

    qt = q[0:n_test, :]
    at = a[0:n_test, :]
    q = q[n_test + 1:, :]
    a = a[n_test + 1:, :]

    print('Number of examples = %d' % (n_exem - n_test))
    step = int(np.around((n_exem - n_test) / num_subsets))
    round_exem = int(step * num_subsets)

    # *************************************************************************
    # Bot training:
    # *************************************************************************

    x = range(0, Epochs)
    valid_loss = np.zeros(Epochs)
    train_loss = np.zeros(Epochs)

    # EOS_index = 1 should be variable because apparently it changes
    for m in range(Epochs):

        # Loop over training batches due to memory constraints:
        for n in range(0, round_exem, step):

            q2 = q[n:n + step]
            s = q2.shape
            count = 0
            for i, sent in enumerate(a[n:n + step]):
                l = np.where(sent == 3)  # the position od the symbol EOS / changed to 1 for current dictionary
                limit = l[0][0]
                count += limit + 1

            Q = np.zeros((count, maxlen_input))
            A = np.zeros((count, maxlen_input))
            Y = np.zeros((count, dictionary_size))

            # Loop over the training examples:
            count = 0
            for i, sent in enumerate(a[n:n + step]):
                ans_partial = np.zeros((1, maxlen_input))

                # Loop over the positions of the current target output (the current output sequence):
                l = np.where(sent == 3)  # the position of the symbol EOS / same here
                limit = l[0][0]

                for k in range(1, limit + 1):
                    # Mapping the target output (the next output word) for one-hot codding:
                    y = np.zeros((1, dictionary_size))
                    y[0, sent[k]] = 1

                    # preparing the partial answer to input:

                    ans_partial[0, -k:] = sent[0:k]

                    # training the model for one epoch using teacher forcing:

                    Q[count, :] = q2[i:i + 1]
                    A[count, :] = ans_partial
                    Y[count, :] = y
                    count += 1

            print('Training epoch: %d, training examples: %d - %d' % (m, n, n + step))
            model.fit([Q, A], Y, batch_size=BatchSize, epochs=1)

            test_input = qt[41:42]
            print(print_result(test_input))
            train_input = q[41:42]
            print(print_result(train_input))
            # prin_result might be wrong and need to change BOS index

        model.save_weights(weights_file, overwrite=True)


# if __name__ == '__main__':

# Getting the data (if you don't have them already or want another room
get_hipchat_room_data_file(room='Berlin')

# For tuning into own data following steps:
# Split the content of your training data into two files: 'context' and 'answers'
split_qa(textfile='data/messages')

# get_train_data.py to store the padded sentences into the files 'Padded_context' and 'Padded_answers';
get_train_data()
# have experimented there..

# Run train_bot.py to train the chatbot
train_bot()

