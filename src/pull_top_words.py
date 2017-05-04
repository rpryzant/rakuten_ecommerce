"""
python pull_top_words.py MORPH_HEALTH/64-True-word_vectors-bahdanau-before_split-0.25-64-64-64/out.pkl ../data/labels/choco.multi_candid.all ../data/large/morph/without_pos/health/vocab TEST_best TEST_worst

TODO: - MULTIPLE ATTENTIONS, TAKE DIF

"""
from collections import defaultdict
import sys
import cPickle
import numpy as np



def load_pickle(fname):
    with open(fname) as f:
        return cPickle.load(f)

def z_scores(l):
    tmp = np.array(l)
    return ((tmp - np.mean(tmp)) / np.std(tmp)).tolist()

def item_ids(labels):
    return list(set([x.strip().split()[1] for x in labels]))

def word_scores_for_ids(output, ids):
    out = []
    for i, id in enumerate(output['ids']):
        if id in ids:
            length = output['len'][i]
            source = output['source'][i][:length]
            scores = z_scores(output['attn'][i][:length])
            out += zip(source, scores)
    return out


def rm_dups(word_scores):
    """ rm dups, keeping each words max score
    """
    tmp = defaultdict(lambda: -100)
    for word, score in word_scores:
        tmp[word] = max(tmp[word], score)
    return tmp.items()


def build_vocab(vocab):
    d = {}
    for i, l in enumerate(vocab):
        d[i + 2] = l.split()[0]      # +2 to reserve 0 for pad, 1 for unk
    d[1] = 'UNK'
    return d


def lookup_words(word_scores, vocab):
    vocab_map = build_vocab(vocab)
    return [(vocab_map[i], s) for i, s in word_scores]    

def get_pth(word_scores, p):
    cutoff = int(len(word_scores) * (1 - p))
    return word_scores[:cutoff]

# load in args
print 'INFO: loading inference output'
inference_output = load_pickle(sys.argv[1])    # keys: ['source', 'ids', 'len', 'attn']
#print inference_output['attn']
labels = open(sys.argv[2])
vocab = open(sys.argv[3])
best_path = sys.argv[4]
worst_path = sys.argv[5]

# extract ids from labels
print 'INFO: extracting labels'
labels = item_ids(labels)

# get (word, score) pairs, remove duplicates, and do index => word lookup
print 'INFO: scoring words'
word_scores = word_scores_for_ids(inference_output, labels)
word_scores = rm_dups(word_scores)
word_scores = lookup_words(word_scores, vocab)

# sort and take the best/worst
print 'INFO: taking best words'
word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
best = get_pth(word_scores, 0.8)
worst = get_pth(word_scores[::-1], 0.8)

# write to file
print 'INFO: writing best words to ', best_path
with open(best_path, 'a') as out:
    for word, score in best:
        out.write(word + '\t' + str(score) + '\n')

print 'INFO: writing worst words to ', worst_path
with open(worst_path, 'a') as out:
    for word, score in worst:
        out.write(word + '\t' + str(score) + '\n')






