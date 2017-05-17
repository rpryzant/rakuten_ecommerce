"""
computes the overlap between two vocab files

python compute_overlap.py [bpe] [morph]

"""
import sys

vocab1 = sys.argv[1]
vocab2 = sys.argv[2]

underscore = '\xe2\x96\x81'
morph = [l.strip().split()[0].replace(underscore, '') for l in open(vocab2)]

def contained(w):
    for mw in morph:
        if w in mw:
            yield w, mw

i = 0
for l in open(vocab1):
    w = l.strip().split()[0].replace(underscore, '')

    matches = [m for m in contained(w)]
    if len(matches) > 0:
        i += 1
print i









