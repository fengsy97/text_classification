#coding:utf-8
#fsy 20190530
#analysis_cnn.py
import warnings
import numpy as np
import time

warnings.filterwarnings('ignore')
STRAT_TIME = time.time()
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
DATA_CLASS = 5
def printUsedTime():
    used_time = time.time() - STRAT_TIME
    print('used time: %.2f seconds' %used_time)

print( '(1) load texts...')

train_label_list = []
train_content_list = []
test_label_list = []
test_content_list = []
with open('../data/train_tag.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()] #类型 文本
    train_label_list = [k.split('\t')[1] for k in line_list]  #类型
    train_content_list = [k.split('\t')[0]for k in line_list]
    # sentences = [s.split() for s in train_content_list]

with open('../data/test_tag.txt', encoding='utf8') as file:
    line_list = [k.strip() for k in file.readlines()] #类型 文本
    test_label_list = [k.split('\t')[1] for k in line_list]  #类型
    test_content_list = [k.split('\t')[0]for k in line_list]
    # sentences = [s.split() for s in train_content_list]

all_texts = train_content_list + test_content_list
all_labels = train_label_list + test_label_list

printUsedTime()

print( '(2) doc to var...')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers


tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)
sequences = tokenizer.texts_to_sequences(all_texts)
word_index = tokenizer.word_index
print(('Found %s unique tokens.' % len(word_index)))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print(('Shape of data tensor:', data.shape))
print(('Shape of label tensor:', labels.shape))

half = int(len(data)*0.8)
#half = len(train_label_list)
x_train = data[:half]
y_train = labels[:half]
x_val = x_train
y_val = y_train
x_test = data[half:]
y_test = labels[half:]
print( 'train docs: '+str(len(x_train)))
print( 'val docs: '+str(len(x_val)))
print( 'test docs: '+str(len(x_test)))
printUsedTime()

print( '(3) training model...')
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding, GlobalMaxPooling1D
from keras.models import Sequential

model = Sequential()
#嵌入层 输入为单词索引表长度、每个词的维度100、每个句子的长度100 
model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(Dropout(0.2))
#100*200
#Conv1D(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None)
model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))

model.add(MaxPooling1D(3))
#将输入展平
model.add(Flatten())
model.add(Dense(EMBEDDING_DIM, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(DATA_CLASS, activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
model.summary()

rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['acc'])
print( model.metrics_names)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=128)
model.save('cnn-rmsprop.h5')
print( '(6) testing model...')
print( model.evaluate(x_test, y_test))