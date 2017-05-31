"""
python correlation.py /Users/rapigan/Documents/rakuten/words_with_mi2/words/rnn/choco/best_bpe_flipped  /Users/rapigan/Documents/rakuten/datasets/bpe/choco/inputs.binary /Users/rapigan/Documents/rakuten/datasets/bpe/choco/outputs.binary
python correlation.py /Users/rapigan/Documents/rakuten/words_with_mi2/words/or/choco/bpe   /Users/rapigan/Documents/rakuten/datasets/bpe/choco/inputs.binary /Users/rapigan/Documents/rakuten/datasets/bpe/choco/outputs.binary
python correlation.py /Users/rapigan/Documents/rakuten/words_with_mi2/words/lasso/choco/choco_bpe_lasso.txt   /Users/rapigan/Documents/rakuten/datasets/bpe/choco/inputs.binary /Users/rapigan/Documents/rakuten/datasets/bpe/choco/outputs.binary
python correlation.py /Users/rapigan/Documents/rakuten/words_with_mi2/words/rnn/choco/best_bpe_notflipped    /Users/rapigan/Documents/rakuten/datasets/bpe/choco/inputs.binary /Users/rapigan/Documents/rakuten/datasets/bpe/choco/outputs.binary


python correlation.py [word_list] [input file] [output file]
"""
import sys
import numpy as np
from tqdm import tqdm

np.warnings.filterwarnings('ignore')


def cramers_v(feature, descriptions, targets, labels):
    """ chisq statistic for a single feature, given some descriptions
        and target info (Y) and target labels (possible values for Y)
    """
    obs = np.zeros( (2, len(labels)) )
    for description, target in zip(descriptions, targets):
        if feature in description:
            obs[1, labels.index(target)] += 1
        else:
            obs[0, labels.index(target)] += 1

    row_totals = np.sum(obs, axis=1)
    col_totals = np.sum(obs, axis=0)
    n = np.sum(obs)
    expected = np.outer(row_totals, col_totals) / n

    chisq = np.sum( np.nan_to_num(((obs - expected) ** 2 ) / expected ))

    r = 2
    k = len(labels)
    phisq = chisq / n
    V = np.sqrt(phisq / min(k-1, r-1))

    return V


def pointwise_biserial(feature, descriptions, prices):
    """ pointwise biserial statistic
    https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient
    """
    s = np.std(prices)

    group0 = []
    group1 = []
    for description, price in zip(descriptions, prices):
        if price == -1:
            continue
        if feature in description:
            group0.append(price)
        else:
            group1.append(price)

    m0 = np.mean(group0)
    m1 = np.mean(group1)

    n0 = float(len(group0))
    n1 = float(len(group1))
    n = n0 + n1

    rpb = (abs(m1 - m0) / s) * np.sqrt((n0 * n1) / (n ** 2))
    if type(rpb) == type(0.0):
        print 'here'
        return None
    return rpb



features = [x.strip().split()[0] for x in open(sys.argv[1])]

descriptions = [x.strip().split() for x in open(sys.argv[2])]

labels = [x.strip().split('|') for x in open(sys.argv[3])]


sales = [int(x[0]) for x in labels]
brands = [x[1] for x in labels]
brand_labels = list(set(brands))
prices = [np.log(float(x[2])) if x[2] != '\N' else -1 for x in labels]

print np.mean([cramers_v(fi, descriptions, brands, brand_labels) for fi in tqdm(features)])
print np.mean(filter(lambda x: not np.isnan(x), [pointwise_biserial(fi, descriptions, prices) for fi in tqdm(features)]))
