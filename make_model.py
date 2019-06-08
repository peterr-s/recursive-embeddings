#!/usr/bin/env python3

import tensorflow as tf

class Word :
    def __init__(self, line) :
        fields = line.split()
        self.lemma = fields[2]
        self.head = int(fields[6])
        self.reln = fields[7]

word_embedding_dim = 300
sent_embedding_dim = 4800
batch_size = 1

# transform word embeddings to sentence dimensionality
t = tf.get_variable(tf.float32, shape = [sent_embedding_dim, word_embedding_dim], name = "t")

# set up x and y
x = tf.placeholder(tf.float32, shape = [sent_embedding_dim + dependency_dim, batch_size], name = "x")
y = tf.placeholder(tf.float32, shape = [sent_embedding_dim, batch_size], name = "y")

# set up dependency embeddings

# set up hidden layer
w = tf.get_variable("w", shape = [hidden_dim, sent_embedding_dim + dependency_dim])

sentences = list()
with open(sys.argv[0], "r") as train_file :
    sentence = list()
    for line in train_file.readlines() :
        if line[0] == "#" :
            continue
        elif line.strip() == "" :
            sentences.append(sentence)
            sentence = list()
        else :
            sentence.append(Word(line))

