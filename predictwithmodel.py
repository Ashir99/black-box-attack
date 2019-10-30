# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:56:00 2019

@author: imran gharib
"""
import os
import numpy as np
import keras
import xlrd
from keras.models import model_from_json
from nltk.corpus import wordnet
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Input, merge, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from keras.layers.merge import Concatenate
from keras.models import Sequential
from nltk.tokenize import word_tokenize, punkt
import pandas as pd
from numpy import genfromtxt, array
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.layers import merge
from keras.layers import Input, Dense, Embedding, merge, Convolution2D, MaxPooling2D, Dropout
from keras.layers.core import Reshape, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import re

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = 98 - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences

#words = ['gold', 'survivor', 'really', 'video', 'hurt', 'full', 'imagination','illusion', 'final', 'true', 'forever', 'cute', 'I','Pink', 'facebook','want','disco','solution','official', 'image','back', 'come','brand','water','flight','display','posted','android', 'auto','song','Club','down','cinema','countdown','young','live','TV','learning','fly','original','download','file','cut','tomatoes','victims','before','wake','big','review','sample','audio','clips','one','terrible','tips','available']
#syn = ['precious', 'residue', 'in real', 'movie', 'harm', 'complete','not real',  'fake', 'last', 'real', 'eternal', 'charming', 'me','color', 'socialmedia','need','dance','answer', 'professional','photograph','front','go','logo','liquid','takeoff','view','published','samrtphone','vehicle','music','pub','bottom','movie hall','launch procedure','youth','alive','television','knowing','takeoff','unique','install','document','trim','vegetable','victimise','preceding','awake','huge','inspect','example','sound','short films','single','disastrous','ideas','obtainable']
words = ['gold','video','survivor','music','eye','facebook','come','clip','culture','club','really','hurt','one','solution','live','final','countdown','official','forever','young','big','audio','cinema','full','song','via','imagination','just','illusion','down','let','true','image','simple','wake','almost','ready','flight','touch','latest','action','victims','monitor','android','young','photo','photos']
syn = ['precious','movie','residue','track','eyeball','socialmedia','approach','pin','civilization','pub','actually','damage','single','mixture','alive','finishing','launch procedure','professional','eternal','youth','great','song','movie arena','filled','melody','per','vision','only','shadow','bottom','pass','real','picture','plain','stay up','nearly','readiness','takeoff','feel','current', 'gesture','victimize','spy','smartphone','youth','snapshot','snapshots']
json_file = open('textModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("textModel.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
"""score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))"""

bot_percent = []
human_percent = []

data= list(open("./test3.txt" ,encoding="latin-1").readlines())
text_data =[s.strip().lower() for s in data]

print(len(text_data))

for i in range(len(text_data)):
    x = text_data[i]
    #print(x)
    x_text = [clean_str(x)] 
    #print (x_text)
    x_text = [s.split(" ") for s in x_text]
#    print(len(x_text[i]))
    #print(x_text)
    """for j in range(len(x_text)):
        for k in range(len(x_text[j])):
            if str(x_text[0][k]).lower() in words:
                ind = words.index(x_text[0][k])
                x_text[0][k] = syn[ind]        
    #print (x_text)
    print(' '.join(x_text[j]))
    text_data[i] = ' '.join(x_text[j])"""
    sentences_padded = pad_sentences(x_text)
    #print (sentences_padded)
    vocabulary = np.load('vocabulary.npy', allow_pickle=True).item()
    #print(type(vocabulary))
    for word in sentences_padded:
        for word2 in word:
#            print(word2)
            try:
                print(vocabulary[word2])
            except:
                vocabulary[word2]=282
                print(vocabulary[word2])
                #print("word not found")
    
    x2 = np.array([[vocabulary[word2] for word2 in word]for word in sentences_padded] )
    print (len(x2))
    #nb_epoch = 10
    #batch_size = 30
 
    """model = load_model('fulldatasetmodel.h5')
    print(model)
    """
    y_pred = loaded_model.predict(x2)
    cc=['Human','Bot']
    for xx in y_pred:
        count=0
        for yy in xx:
            if count == 0:
                human = str(format(yy*100,'.2f'))
                bot_percent.append(yy*100)
                print (cc[count] +" " + human + "%")
            if count == 1:
                bot = str(format(yy*100,'.2f'))
                human_percent.append(yy*100)
                print (cc[count] +" " + bot + "%")
            #f.write(cc[count] +", " + str(format(yy*100,'.2f')) + "%")
            count = count + 1

f = open('test3.csv','w', encoding="utf-8")
f.write('tweets, Human%, Bot% \n')
for i in range(len(text_data)):
    f.write('%s, %0.2f, %0.2f \n' %(text_data[i], bot_percent[i], human_percent[i]))
    #Give your csv text here.

avg1 = sum(bot_percent)/len(bot_percent)
avg2 = sum(human_percent)/len(human_percent)
f.write(', %0.2f, %0.2f \n' %(avg1, avg2))
print('Total Accuracy\n')
human = 'Human = %0.2f' %avg1
print(human + '%\n')
bot = 'Bot = %0.2f' %avg2
print(bot + '%')
f.close()

