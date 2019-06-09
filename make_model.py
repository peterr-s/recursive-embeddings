#!/usr/bin/env python3

import sys
import tensorflow as tf
import numpy as np
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import skipthoughts

class Word :
    def __init__(self, line) :
        fields = line.split()
        self.surface = fields[1]
        self.parent = int(fields[6]) if fields[6].isdigit() else 0
        self.reln = fields[7]
        self.children = set()

    def __str__(self) :
        return self.surface

sent_embedding_dim = 4800
hidden_size = 1000
max_sent_len = 15
learning_rate = 0.15
epoch_ct = 200

# load training sentences from input file
sentences = list()
with open(sys.argv[1], "r") as train_file :
    sentence = list()

    for line in train_file.readlines() :
        if line[0] == "#" : # ignore commented lines
            continue
        elif line.strip() == "" : # blank lines denote a sentence boundary
            if len(sentence) <= max_sent_len :
                # associate each word except the root with its children
                for i in range(1, len(sentence)) :
                    sentence[sentence[i].parent].children.add(sentence[i])
                # add to training set
                sentences.append(sentence)

            sentence = [Word("0 %ROOT %ROOT - - - 0 %ROOT - -")]
        else :
            sentence.append(Word(line))

    if not sentences[-1] == sentence : # guard against lack of trailing newline
        sentences.append(sentence)

# load embeddings
embeddings = KeyedVectors.load_word2vec_format(sys.argv[3], binary = False)
word_embedding_dim = embeddings.vector_size
null_word = np.zeros(word_embedding_dim)
#null_word = np.random.rand(word_embedding_dim)

skipthoughts_model = skipthoughts.load_model()
skipthoughts_encoder = skipthoughts.Encoder(skipthoughts_model)

sentence_embeddings = skipthoughts_encoder.encode([" ".join([w.surface for w in s]) for s in sentences])
batch_size = sentence_embeddings.shape[0]

# set up target
y = tf.placeholder(dtype = tf.float32, shape = [sent_embedding_dim, batch_size], name = "y")

# transform word embeddings to sentence dimensionality
t = tf.get_variable("t", dtype = tf.float32, shape = [sent_embedding_dim, word_embedding_dim])

# recursive graph building
def compose_embedding(word) :
    with tf.variable_scope("compose", reuse = tf.AUTO_REUSE) :
        x = tf.Variable(
                tf.transpose(
                    [list(embeddings[word.surface] if word.surface in embeddings else null_word)]
                    ),
                trainable = False
                ) if word.surface is not "%ROOT" else tf.get_variable("x_ROOT") # make the embedding untrainable EXCEPT for root
        x = tf.matmul(t, x)
        for child in word.children :
            w1 = tf.get_variable(
                    ("w1_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [hidden_size, x.shape[0]]
                    )
            b1 = tf.get_variable(
                    ("b1_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [hidden_size, 1]
                    )
            w2 = tf.get_variable(
                    ("w2_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [sent_embedding_dim, hidden_size]
                    )
            b2 = tf.get_variable(
                    ("b2_%s" % child.reln),
                    dtype = tf.float32,
                    shape = [sent_embedding_dim, 1]
                    )
            
            x = tf.math.l2_normalize(
                        tf.nn.leaky_relu(
                        tf.matmul(
                            w2,
                            tf.nn.leaky_relu(
                                tf.matmul(
                                    w1,
                                    x
                                    ) + b1
                                )
                            ) + b2
                        )
                    )

        return x

if __name__ == "__main__" :
    sess = tf.Session()

    # for each sentence, build a graph and then do a round of training
    preds = list()
    for i, sentence in enumerate(sentences) :
        preds.append(compose_embedding(sentence[0]))
    y_pred = tf.concat(preds, axis = 1)

    with tf.variable_scope("train", reuse = tf.AUTO_REUSE) :
        loss = tf.losses.cosine_distance(tf.math.l2_normalize(y), y_pred, axis = 1)
        loss = tf.identity(loss, name = "loss")
        train = tf.train.AdamOptimizer(learning_rate).minimize(loss, name = "train")

        sess.run(tf.global_variables_initializer())
        for _ in range(epoch_ct) :
                l_val = sess.run([loss, train], {y: np.transpose(sentence_embeddings)})
                print("loss:", l_val)

