# compose-embeddings

#### UW LING 575 final project

This project is designed to find generalizable embedding composition functions. The objective is to have word embeddings and a dependency parse as input, with an established sentence embedding as the target output.

### Dependencies

* tensorflow
* theano
* numpy
* scipy
* sklearn
* nltk
* gensim

Included is a copy of [tartarskunk's fork of skipthoughts](https://github.com/tartarskunk/skip-thoughts) because I don't know a better way to work with Python modules.

This project currently runs on Python, but will eventually be converted to use as much Rust as possible.

