"""
generates vocab from tokenized files

python generate_vocab.py [text] > [out]


python data_wrangling/generate_vocab.py data/morph_small/total.inputs > data/morph_small/vocab
"""
import sys


d = set()

for l in open(sys.argv[1]):
    d.update(l.strip().split())

for x in d:
    print x
