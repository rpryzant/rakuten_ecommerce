"""
compute the mutual information for each token in a lexicon


python mutual_information.py [vocab] [binary labels] [source]
"""
import sys
import math



def marginal(target, w, labels, source):
    count = sum([w in desc.split() for (desc, lab) in zip(source, labels) if lab == target])
    return float(count) / len(source)

def joint(w, source):
    count = sum([w in desc.split() for desc in source])
    return float(count) / len(source)


vocab = [l.strip().split()[0] for l in open(sys.argv[1])]
labels = [l.strip().split('|')[0] for l in open(sys.argv[2])]
source = [l.strip() for l in open(sys.argv[3])]


for w in vocab:
    marg_0 = marginal('0', w, labels, source)
    marg_1 = marginal('1', w, labels, source)
    joint_prob = joint(w, source)
    mi = joint_prob * math.log(joint_prob / (marg_0 * marg_1))
    print '%s\t%s' % (w, mi)
