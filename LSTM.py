from keras.layers.core import Activation, Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
# from keras.preprocessing import sequence
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import re

def data_cleaning(text):
    text=re.sub('<[^>]*>','',text)
    emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower()) + ' '.join(emojis).replace('-','')
    # text = re.sub('[^a-zA-Z\s]', " ", text)
    return text

def clear_puntuation(tokens):
    import string
    filtered_tokens = [token for token in tokens if token not in string.punctuation]
    return filtered_tokens

import nltk
from nltk.corpus import stopwords
stopword_list = stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("not")

maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
with open('./train_ds.txt','r+', encoding='UTF-8') as f:
    for line in f.readlines():
        label, sentence = line[0], line[2:-1]
        sentence = data_cleaning(sentence)
        words = nltk.word_tokenize(sentence.lower())
        words = [word for word in words if word not in stopword_list]
        clear_puntuation(words)
        # words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))
# max_feature for nb_word / max_sentence_len for max len
MAX_FEATURES = 30000
MAX_SENTENCE_LENGTH = 250
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2
word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word_index["PAD"] = 0
word_index["UNK"] = 1
# transform the word_index into the txt format
# with open('word2index.txt', 'w', encoding="utf-8") as file:
#     for word, num in word_index.items():
#         file.write(word + "\t" + str(num) + "\n")

# index2word = {v:k for k, v in word_index.items()}
X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
# print(word_index)
i=0
# use the dictionary to save the words
with open("train_ds.txt", 'r', encoding = 'utf-8') as file:
    for line in file.readlines():
        label, sentence = line[0], line[2:-1]
        # words = nltk.word_tokenize(sentence.lower())
        sentence = data_cleaning(sentence)
        words = nltk.word_tokenize(sentence.lower())
        words = [word for word in words if word not in stopword_list]
        clear_puntuation(words)
        seqs = []
        for word in words:
            if word in word_index:
                seqs.append(word_index[word])
            else:
                seqs.append(word_index["UNK"])
        X[i] = seqs
        y[i] = int(label)
        i += 1
    
X = pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# data split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
# construct the model
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
BATCH_SIZE = 32
NUM_EPOCHS = 3
model = Sequential()
# add embedding layer
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
# add lstm layer
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model.summary()
# train the model
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
print("\nValidation score: %.3f, accuracy: %.3f" % (score, acc))
    
# save the model
model.save('Sentiment1.h5')  # creates a HDF5 file 'model.h5'

X_pred = np.empty(num_recs,dtype=list)
y_pred = np.zeros(num_recs)
i=0
# use the dictionary to save the words
with open("test_ds.txt", 'r', encoding = 'utf-8') as file:
    for line in file.readlines():
        label, sentence = line[0], line[2:-1]
        # words = nltk.word_tokenize(sentence.lower())
        sentence = data_cleaning(sentence)
        words = nltk.word_tokenize(sentence.lower())
        words = [word for word in words if word not in stopword_list]
        clear_puntuation(words)
        seqs = []
        for word in words:
            if word in word_index:
                seqs.append(word_index[word])
            else:
                seqs.append(word_index["UNK"])
        X_pred[i] = seqs
        y_pred[i] = int(label)
        i += 1
X_pred = pad_sequences(X_pred, maxlen=MAX_SENTENCE_LENGTH)

score, acc = model.evaluate(X_pred, y_pred)
print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
