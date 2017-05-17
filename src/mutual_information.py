"""
compute the mutual information for each token in a lexicon


python mutual_information.py [vocab] [binary labels] [source]
"""
import sys
import math
import os
from joblib import Parallel, delayed

#                                         bpe/choco
OUTPUTS = '/Users/rapigan/Desktop/datasets/%s/%s/outputs.binary'
INPUTS = '/Users/rapigan/Desktop/datasets/%s/%s/inputs.binary'




def marginal(target, w, labels, source):
    count = sum([w in desc.split() for (desc, lab) in zip(source, labels) if lab == target])
    return float(count) / len(source)

def joint(w, source):
    count = sum([w in desc.split() for desc in source])
    return float(count) / len(source)


def gen_mi(vocab_src):
    token_type = 'bpe' if 'bpe' in vocab_src else 'morph'
    data_type = 'health' if 'health' in vocab_src else 'choco'

    vocab = [l.strip().split()[0] for l in open(vocab_src)]
    labels = [l.strip().split('|')[0] for l in open(OUTPUTS % (token_type, data_type))]
    source = [l.strip() for l in open(INPUTS % (token_type, data_type))]

    out = ''
    for w in vocab:
        marg_0 = marginal('0', w, labels, source)
        marg_1 = marginal('1', w, labels, source)
        joint_prob = joint(w, source)
        try:
            mi = joint_prob * math.log(joint_prob / (marg_0 * marg_1))
        except:
            mi = -1
        out += '%s\t%s\n' % (w, mi)
    
    with open(vocab_src + '.mi', 'w') as f:
        f.write(out)
    print 'done! ', vocab_src


vocabs = [
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/choco/choco_bpe_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/choco/choco_morph_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/choco/mi_choco_bpe_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/health/health_bpe_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/health/health_morph_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/lasso/health/mi_health_bpe_lasso.txt',
    '/Users/rapigan/Desktop/words_with_mi2//words/or/choco/bpe',
    '/Users/rapigan/Desktop/words_with_mi2//words/or/choco/morph',
    '/Users/rapigan/Desktop/words_with_mi2//words/or/health/bpe',
    '/Users/rapigan/Desktop/words_with_mi2//words/or/health/morph',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_bpe_flipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_bpe_notflipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_morph_flipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/choco/best_morph_notflipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_bpe_flipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_bpe_notflipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_morph_flipped',
    '/Users/rapigan/Desktop/words_with_mi2//words/rnn/health/best_morph_notflipped'
]
Parallel(n_jobs=4)(delayed(gen_mi)(v) for v in vocabs)
