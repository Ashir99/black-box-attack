import os
import numpy as np
import keras
import tensorflow as tf

from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input,Convolution2D, MaxPooling2D
from data_helpers import load_data
from keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.layers.merge import Concatenate
from keras.models import Sequential
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Reshape, Flatten
from sklearn.model_selection import train_test_split


x, y, vocabulary, vocabulary_inv = load_data()
text= pd.DataFrame(x)


sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 256
filter_sizes = [3,4,5]
num_filters = 512
drop = 0.5

inputs1 = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(output_dim=embedding_dim, input_dim=vocabulary_size, input_length=sequence_length)(inputs1)
reshape = Reshape((sequence_length,embedding_dim,1))(embedding)

conv_0 = Convolution2D(num_filters, filter_sizes[0], embedding_dim, border_mode='valid', kernel_initializer='normal', activation='relu', dim_ordering='tf')(reshape)
conv_1 = Convolution2D(num_filters, filter_sizes[1], embedding_dim, border_mode='valid', kernel_initializer='normal', activation='relu', dim_ordering='tf')(reshape)
conv_2 = Convolution2D(num_filters, filter_sizes[2], embedding_dim, border_mode='valid', kernel_initializer='normal', activation='relu', dim_ordering='tf')(reshape)
# paralle conv and pool layer which process each section of input independently
#conv1 = Convolution2D(512, (3, 3), activation='relu')(inp1)
maxp1 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_0)
maxp2 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_1)
maxp3 = MaxPooling2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), border_mode='valid', dim_ordering='tf')(conv_2)

# can add multiple parallel conv, pool layes to reduce size
conv_blocks = []
conv_blocks.append(maxp1)
conv_blocks.append(maxp2)
conv_blocks.append(maxp3)
z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
flt1 = Flatten()(z)
dropout = Dropout(drop)(flt1)
#flt2 = Flatten()(conv2)
output = Dense(2, activation='softmax')(dropout)

np.save('vocabulary.npy',vocabulary)

model = Model(input=inputs1, output=output)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)



model.fit(X_train, y_train, epochs=3, batch_size=32)

model_json=model.to_json()
with open("textModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("textModel.h5")




