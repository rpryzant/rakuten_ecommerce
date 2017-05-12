# -*- coding: utf-8 -*-
"""
python plot_pos_category_hist.py /Users/rapigan/Desktop/choco/en/morph_best  /Users/rapigan/Desktop/choco/ja/morph_best morph_best

"""
import sys
import os
import string
from nltk.corpus import wordnet as wn
import re
from collections import Counter
import nltk 
from geotext import GeoText
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#import enchant
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import numpy as np



# from https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# non-content types have been commented out
PENN_POS_TAGS = {
    'CC': 'COORDINATING CONJUNCTION',
    'CD': 'CARDINAL NUMBER',
    'DT': 'DETERMINER',
    'EX': 'EXISTENTIAL THERE',
    'FW': 'FOREIGN WORD',
    'IN': 'PREPOSITION OR SUBORDINATING CONJUNCTION',
    'JJ': 'ADJECTIVE',
    'JJR': 'ADJECTIVE, COMPARATIVE',
    'JJS': 'ADJECTIVE, SUPERLATIVE',
    'MD': 'MODAL',
    'NN': 'NOUN, SINGULAR OR MASS',
    'NNS': 'NOUN, PLURAL',
    'NNP': 'PROPER NOUN, SINMGULAR',
    'NNPS': 'PROPER NOUN, PLURAL',
    'PDT': 'PREDETERMINER',
    'POS': 'POSSESSIVE ENDING',
    'PRP': 'PERSONNNAL PRONOUN',
    'PRP$': 'POSSESSIVE PRONOUN',
    'RB': 'ADVERB',
    'RBR': 'ADVERB, COMPARATIVE',
    'RBS': 'ADVERB, SUPERLATIVE',
    'RP': 'PARTICLE',
    'SYM': 'SYMBOL',
    'TO': 'TO',
    'UH': 'INTERJECTION',
    'VB': 'VERB BASE',
    'VBD': 'VERB PAST TENSE',
    'VBG': 'VERB GERUND OR PRESENT PARTICIPLE',
    'VBN': 'VERB PAST PARTICIPLE',
    'VBP': 'VERB NON-3RD PERSON SINGULAR PRESENT',
    'VBZ': 'VERB 3RD PERSON SINGULAR PRESENT',
    'WDT': 'WH-DETERMINER',
    'WP': 'WP-PRONOUN',
    'WP$': 'POSSESSIVE WH-PRONOUN',
    'WRB': 'WH-ADVERB'
}

CONTENT_POS_TAGS = [
    'CD', 'FW', 'JJ', 'JJR', 'JJS', 'NN',
    'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
]








def jaccard(words1, words2):
    # TODO
    pass


def sentiment(word, analyzer):
    polarity = analyzer.polarity_scores(word)
    if polarity['neg']:
        return 'neg'
    elif polarity['pos']:
        return 'pos'
    else:
        return 'neu'


def full_word(word):
    # TODO
    pass


def char_type(word):
    def is_hiragana(c):
        return ord(u'ぁ') <= ord(c) <= ord(u'ゟ')

    def is_katakana(c):
        return ord(u'゠') <= ord(c) <= ord(u'ヿ')

    def is_kanji(c):
        return ord(u'一') <= ord(c) <= ord(u'龯') or \
                ord(u'㐀') <= ord(c) <= ord(u'䶵')

    def is_romanji(c):
        return ord(u'０') <= ord(c) <= ord(u'ｚ') or \
                ord('A') <= ord(c) <= ord('z')

    # hir/kat/kanji/rom
    hiragana_chars = [is_hiragana(c) for c in word.decode('utf8')]
    katakana_chars = [is_katakana(c) for c in word.decode('utf8')]
    kanji_chars =  [is_kanji(c) for c in word.decode('utf8')]
    romanji_chars =  [is_romanji(c) for c in word.decode('utf8')]

    if any(kanji_chars):
        return 'kanji'
    elif any(hiragana_chars) and sum(hiragana_chars) > sum(katakana_chars):
        return 'hiragana'
    elif any(katakana_chars):
        return 'katakana'
    elif any(romanji_chars):
        return 'romanji'
    else:
        return 'other'


def is_loc(word):
    # TODO
    gt = GeoText(word)
    return len(gt.cities) + len(gt.country_mentions) > 0 # both have to be geos


def length(word):
    return len(word)


def pos(word):
    try:
        return nltk.pos_tag([word])[0][1]
    except:
        return None


def hypernym(word, height=3):
    try:
        best = None
        for i in range(height):
            syns = wn.synsets(word)[0]
            hyp = str(syns.hypernyms()[0])
            word = re.findall("'(\w+)", hyp)[0]
            best = word, i
        return best
    except:
        if best is not None:
            return best
        return  word, -1

def rm_punctuation(w):
    return ''.join([c for c in w if c not in string.punctuation])


def gen_words(line, raw_ja, STOPS):
    def stop_word(w, STOPS):
        return rm_punctuation(w.lower()) in STOPS

    if char_type(raw_ja) == 'other':
        yield raw_ja
        return

    for x in line.split()[:-1]:
        if not stop_word(x, STOPS):
            yield x


def continuous_hist(data, xlab='values'):
    xmin = min(d1 + d2)
    xmax = max(d1 + d2)

    bins = np.linspace(xmin, xmax, 100)

    plt.hist(data, bins, alpha=0.5, label=xlab)
    plt.legend(loc='upper right')
    plt.show()



def categorical_hist(data, title='title', name=None, xlab='categories'):
    data = Counter(data)

    counts = []
    labs = []
    for k, v in data.items():
        if v > 10:
            labs.append(k)
            counts.append(v)

    fig, ax = plt.subplots()
    width = 0.4
    xrng = range(len(counts))

    ax.bar(xrng, counts, width)
    ax.set_xticks([i+width/2 for i in xrng])
    ax.set_xticklabels(labs, rotation=45)

    if name is not None:
        fig.savefig("%s.png" % name)
    else:
        plt.show()






NA = 'N/A'
STOPS = [rm_punctuation(w.strip().lower()) for w in open('stop_words.txt')]
en = open(sys.argv[1])
ja = open(sys.argv[2])
plt_name = sys.argv[3]
analyzer = SentimentIntensityAnalyzer()
#en_dict = enchant.Dict("en_US")

word_info = {}

hyps = []
poss = []
locs = []
types = [] 
sents = []
fulls = []
for ja_line, line in zip(ja, en):
    ja_word = ja_line.split()[0]
    for word in gen_words(line, ja_word, STOPS):
        types.append(char_type(ja_word))

        if types[-1] != 'other':
            hyp, height = hypernym(word, height=7)
            hyps.append(hyp if height > -1 else 'N/A')
            poss.append(pos(word))
            locs.append(is_loc(word))
            sents.append(sentiment(word, analyzer))
            fulls.append(full_word(ja_word))
        else:
            hyps.append(NA)
            poss.append(NA)
            locs.append(NA)
            sents.append(NA)
            fulls.append(NA)


#        print word, ja_word, hyp_w, pos_w, loc_w, char_w, sent_w, word_w

categorical_hist(hyps, 'hyponyms', name=plt_name + '_hyponyms')
categorical_hist(types, 'word types', name=plt_name + '_types')
categorical_hist(poss, 'pos', name=plt_name + '_pos')
categorical_hist(locs, 'location?', name=plt_name + '_locations')
categorical_hist(sents, 'sentiment', name=plt_name + '_sentiment')


print len(Counter([]))
