"""
compute the mutual information for each token in a lexicon


python mutual_information.py [vocab] [binary labels] [source]
"""
import sys
import math
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import Counter
#                                         bpe/choco
OUTPUTS = '/Users/rapigan/Desktop/datasets/%s/%s/outputs.binary'
INPUTS = '/Users/rapigan/Desktop/datasets/%s/%s/inputs.binary'

def mutual_information(w, labels, source):
    n00 = 1.  # docs without term, 0 label
    n01 = 1.  # docs without term, 1 label
    n10 = 1.  # docs with term, 0 label
    n11 = 1.  # docs with term, 1 label
    for (desc_ctr, lab) in zip(source, labels):
        if lab == '0':
            if w in desc_ctr:
                n10 += 1
            else:
                n00 += 1
        else:
            if w in desc_ctr:
                n11 += 1
            else:
                n01 += 1
    n0_ = n00 + n01   # docs without term
    n1_ = n11 + n10   # docs with term
    n_0 = n10 + n00   # docs with 0 label
    n_1 = n11 + n01   # docs with 1 label
    n = n00 + n01 + n11 + n10   # total n    

    mutual_info = (n11/n) * math.log((n * n11) / (n1_ * n_1)) + \
                  (n01/n) * math.log((n * n01) / (n0_ * n_1)) + \
                  (n10/n) * math.log((n * n10) / (n1_ * n_0)) + \
                  (n00/n) * math.log((n * n00) / (n0_ * n_0))

    return mutual_info



def gen_mi(vocab_src):
    token_type = 'bpe' if 'bpe' in vocab_src else 'morph'
    data_type = 'health' if 'health' in vocab_src else 'choco'

    vocab = [l.strip().split()[0] for l in open(vocab_src)]
    labels = [l.strip().split('|')[0] for l in open(OUTPUTS % (token_type, data_type))]
    source = [Counter(l.strip().split()) for l in open(INPUTS % (token_type, data_type))]

    out = []
    for w in tqdm(vocab):
        mi = mutual_information(w, labels, source)
        out.append((mi, w))

    out_s = '\n'.join('%s\t%s' % (w, mi) for mi, w in sorted(out, reverse=True))


    with open(vocab_src + '.mi', 'w') as f:
        f.write(out_s)
    print 'done! ', vocab_src


#vocabs = [
#    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/choco/choco_bpe_lasso.txt',
#    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/choco/choco_morph_lasso.txt',
#    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/health/health_bpe_lasso.txt',
#    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/health/health_morph_lasso.txt',
#    '/Users/rapigan/Desktop/words_with_mi2//words/or/choco/bpe',
#    '/Users/rapigan/Desktop/words_with_mi2//words/or/choco/morph',
#    '/Users/rapigan/Desktop/words_with_mi2//words/or/health/bpe',
#    '/Users/rapigan/Desktop/words_with_mi2//words/or/health/morph',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_bpe_flipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_bpe_notflipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_morph_flipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_morph_notflipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_bpe_flipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_bpe_notflipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_morph_flipped',
#    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_morph_notflipped'
#]

vocabs = [
    '/Users/rapigan/Desktop/vocabs/bpe/choco/vocab',
    '/Users/rapigan/Desktop/vocabs/bpe/health/vocab',
    '/Users/rapigan/Desktop/vocabs/morph/choco/vocab',
    '/Users/rapigan/Desktop/vocabs/morph/health/vocab'
]
#gen_mi(vocabs[-1])




Parallel(n_jobs=4)(delayed(gen_mi)(v) for v in vocabs)
