"""
TODO --- REDO THIS SCRIPT


python pull_random_words.py rnn_states-bahdanau-large/out.pkl \
                            ../data/labels/health_multi_candid3.txt \
                            ../data_wrangling/bpe.vocab \
                            health-random_baseline
"""
from collections import defaultdict
import sys
import cPickle
import numpy as np
import random

def build_vocab(vocab):
    d = {}
    for i, l in enumerate(vocab):
        d[i + 2] = l.split()[0]      # +2 to reserve 0 for pad, 1 for unk
    d[1] = 'UNK'
    return d

def get_pth(words, p):
    cutoff = int(len(words) * (1 - p))
    return words[:cutoff]





# load in args
inference_output = sys.argv[1]
labels = open(sys.argv[2])
vocab = open(sys.argv[3])
out_path = sys.argv[4]



vocab = build_vocab(vocab)
words = vocab.values()
random.shuffle(words)


cutoff = int(len(words) * (1 - 0.8))
with open(out_path, 'a') as out:
    for i in range(cutoff):
        out.write(words[i] + '\t' + '0.0' + '\n')
