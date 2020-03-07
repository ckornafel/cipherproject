#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Feb 25 23:54:53 2020

@author: C Kornafel
Attempting to create a translation dictionary of plain text and cipher text. 
This example will be using known plaintext/ciphertext relationships - however a similar
dataset could be executed given the relative frequencies of each term in the text 
and matching those that are close. While the former method would likley reduce the accuracy of 
the "translation", it may offer enough correct terms to predict corresponding text relationships
using by measureing their similarities. 


Following LSTM set-up from: https://stackabuse.com/python-for-nlp-neural-machine-translation-with-seq2seq-in-keras/
Usman Malik, 2019
"""
#Data processing packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model

#Text processing packages
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


from string import digits
import re

#Neural Network model packages
from keras.models import Model
from keras.layers import GRU, Embedding, Dense, LSTM, Input

import tensorflow as tf



#Loading the data
#I had issues with using the keras NN with a pandas dataframe of text - I infer that the pd format
#does not have the required end of line / start of line characters

#Creating lists to store the text
plain_txt = [] #To hold the plain text
cipher_txt = [] #To hold the cipher text
cipher_txt_in = [] #To hold the start of text sequences
   

count = 0
for line in open(r'/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/corp.txt',encoding="utf-8"):
 
    
    #Adding a count only load a partial dataset due to memory issues
    count +=1
    if count > 20000:
        break
    if '\t' not in line:
        continue
    
    p_txt,  output = line.rstrip().split('\t')
    
    
    if len(p_txt)<50:
        p_txt  = p_txt.strip('\"')
        #p_txt = re.sub("'", '', p_txt)
        #p_txt = re.sub(r'[^\w\s]','',p_txt)
        #p_txt = p_txt.rstrip().strip()
        
        plain_txt.append(p_txt)
    
    if len(output) < 50:
        output = output.strip('\"')
        #output = re.sub("'", '', output)
        #output = re.sub(r'[^\w\s]','',output)
        #output = output.rstrip().strip()
        c_txt = output + ' <eos>'
        c_txt_in = '<sos> ' + output
        
        cipher_txt.append(c_txt)
        cipher_txt_in.append(c_txt_in)

    
    



     

#Checking successful load of data
print("plaintext lines loaded: ", len(plain_txt))
print("ciphertext lines loaded: ", len(cipher_txt))
print("ciphertext in lines loaded: ", len(cipher_txt_in))
     
cipher_txt[5]
#Tokenize the text into its base components

#Found a memory issue and discovered that it has a max of 20,000 words
p_token = Tokenizer(num_words = 20000)
c_token = Tokenizer(num_words = 20000, filters = '')

p_token.fit_on_texts(plain_txt)
c_token.fit_on_texts(cipher_txt + cipher_txt_in)

#Transforming the tokens into sequences
p_int_seq = p_token.texts_to_sequences(plain_txt)
c_int_seq = c_token.texts_to_sequences(cipher_txt)
cin_int_seq = c_token.texts_to_sequences(cipher_txt_in)

#Creating a word index
p_word2idx = p_token.word_index
c_word2idx = c_token.word_index

print('Total Unique Words in plaintext: %s' % len(p_word2idx))
print('Total Unique Words in ciphertext: %s' % len(c_word2idx))
#Given that the cipher for level 1 is a rotating cipher, it is expected that there would
#be more cipher words than plain ones since the same plain word could have multiple iterations
#of its encrypted counterpart


longest_plaintext = max(len(sen) for sen in p_int_seq)
num_wds_cipher = len(c_word2idx)+1
longest_ciphertext = max(len(sen) for sen in c_int_seq)

print("Length of longest string in plaintext: %g" % longest_plaintext)
print("Length of longest string in ciphertext: %g" % longest_ciphertext)
#It appears that the tokenizer is able to recognize multiple word break types (inc. punct.). 
#This would account for the larger amount of unique words than I found in R

#Padding: LSTM requires all strings to be the same size to take into account positions relative to each other
#Ironic that I removed the test1 padding only to put it back in
p_encod_seq = pad_sequences(p_int_seq, maxlen = longest_plaintext)
c_encod_seq = pad_sequences(c_int_seq, maxlen = longest_ciphertext, padding='post')


#One Hot Encoding vectors take up space that is unneccessary. 
#Keras offers a dense vector n-dimensional word encodings instead
#embedding_lyr = Embedding( [vocab size or num unique words], [num of dim for each vector], input_length = [length of sentence] )

#Citing GloVe
#Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf] [bib]

#Using prebuild encoding vecotrso from GloVe
from numpy import array
from numpy import asarray
from numpy import zeros

embed_dict = dict()
g = open(r'/Users/ckornafel/Desktop/MSDS692 Data Science Practicum I/glove.6b.100d.txt', encoding = "utf8")

for line in g:
    rec = line.split()
    word = rec[0]
    vec_dim = asarray(rec[1:], dtype='float32')
    embed_dict[word] = vec_dim
g.close()

n_wds = min(20000, len(p_word2idx)+1)
embed_mat = zeros((n_wds, 100))
for word, index in p_word2idx.items():
    embed_vec = embed_dict.get(word)
    if embed_vec is not None:
        embed_mat[index]= embed_vec

print(embed_mat[79])

embed_layer = Embedding(n_wds, 100, weights = [embed_mat], input_length=longest_plaintext)

#Encoding shape: (number of plaintext[75], length of ciphertext term [76385], number of words in ciphertext [73])
#Creating the array to hold the cipher encoding
decipher_target_onehot = np.zeros((
        len(plain_txt),
        longest_ciphertext,
        num_wds_cipher),
    dtype='float32')

#Checking to see if the values are as expected    
decipher_target_onehot.shape #match!

#inserting integers in the place of certain words
for i, d in enumerate(c_encod_seq):
    for t, word in enumerate(d):
        decipher_target_onehot[i, t, word] = 1


#Defining the encoder/decoder of the LSTM
encod_place = Input(shape=(longest_plaintext,))
x = embed_layer(encod_place)
encod = LSTM(256, return_state= True)

encod_out, h, c = encod(x)
encod_states = [h,c]

decipher_place = Input(shape=(longest_ciphertext,))
decipher_embedding = Embedding(num_wds_cipher, 256)
decipher_input_x = decipher_embedding(decipher_place)

decipher_lstm = LSTM(256, return_sequences=True, return_state=True)
decipher_outputs, _, _ = decipher_lstm(decipher_input_x, initial_state=encod_states)

#Predict output
decipher_dense = Dense(num_wds_cipher, activation='softmax')
decipher_outputs = decipher_dense(decipher_outputs)

#Complile Model
mod = Model([encod_place,decipher_place],decipher_outputs)
mod.compile(
        optimizer = 'rmsprop',
        loss = 'categorical_crossentropy',
        metrics=['accuracy'])


plot_model(mod, to_file='ciphermod3.png', show_shapes = True, show_layer_names = True)

 
c_mod = mod.fit(
    [p_encod_seq, c_encod_seq],
    decipher_target_onehot,
    batch_size = 50,
    epochs = 10,
    validation_split = 0.1,)


##############################

encoder_model = Model(encod_place, encod_states)

decipher_state_input_h = Input(shape=(256,))
decipher_state_input_c = Input(shape=(256,))
decipher_state_inputs = [decipher_state_input_h, decipher_state_input_c]

decipher_inputs_single = Input(shape=(1,))
decipher_inputs_single_x = decipher_embedding(decipher_inputs_single)

decipher_outputs, h, c = decipher_lstm(decipher_inputs_single_x, initial_state=decipher_state_inputs)

decipher_states = [h,c]
decipher_outputs = decipher_dense(decipher_outputs)

decipher_model = Model( [decipher_inputs_single] + decipher_state_inputs, 
                       [decipher_outputs]+decipher_states)

plot_model(decipher_model, to_file='decoding_model.png', show_shapes=True, show_layer_names=True)


p_idx2word = {v:k for k, v in p_word2idx.items()}
c_idx2word = {v:k for k, v in c_word2idx.items()}


def decipher_term(plain):
    
    states_value = encoder_model.predict(plain)
    cipher_term = np.zeros((1,1))
    cipher_term[0, 0] = c_word2idx['<sos>']
    eos = c_word2idx['<eos>']
    deciphered = []
    
    for x in range(longest_ciphertext):
        cipher_token, h, c = decipher_model.predict([cipher_term]+states_value)
        idx = np.argmax(cipher_token[0,0,:])
        
        if eos == idx:
            break
        
        word = ' '
        
        if idx > 0:
            word = c_idx2word[idx]
            deciphered.append(word)
            
        cipher_term[0,0] = idx
        states_value = [h,c]
        
    return ' '.join(deciphered)
        

i = np.random.choice(len(plain_txt))


plain = p_encod_seq[i:i+1]
decipher = decipher_term(plain)

print("Plaintext: ", plain_txt[i])
print("Ciphertext: ", decipher)  

    states_value = encoder_model.predict(plain)
    cipher_term = np.zeros((1,1))
    cipher_term[0, 0] = c_word2idx['<sos>']
    eos = c_word2idx['<eos>']
    deciphered = []
    
    for x in range(longest_ciphertext):
        cipher_token, h, c = decipher_model.predict([cipher_term]+states_value)
        idx = np.argmax(cipher_token[0,0,:])
        print(idx)
        if eos == idx:
            break
        
        word = ' '
        
        if idx > 0:
            word = c_idx2word[idx]
            deciphered.append(word)
            
        cipher_term[0,0] = idx
        states_value = [h,c]



        

    











