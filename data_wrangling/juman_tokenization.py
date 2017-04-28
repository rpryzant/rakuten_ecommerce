"""
Tokenizes an input corpus file (one japanese text sequence per line)
    with the Juman tokenizer 
For more infor on Juman: http://www.utali.io/entry/2016/10/20/202744

https://pypi.python.org/pypi/JapaneseTokenizer

"""
import argparse
import json
from collections import Counter
import os
import commands
from tqdm import tqdm

from pyknp import Jumanpp

from jpn.deinflect import guess_stem
from rakutenma import RakutenMA


def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description='Usage') # add description
    # positional arguments
    parser.add_argument('corpus', metavar='corpus', type=str, help='input corpus file (JA)')
    parser.add_argument('output', metavar='output', type=str, help='where to write tokenized outputs')

    # optional arguments
    parser.add_argument('-pos', '--pos', dest='pos', action='store_true', help='create word:pos features')
    parser.add_argument('-v', '--vocab', dest='vocab', type=str, default=None, help='optional vocab output')
    parser.add_argument('-t', '--threads', dest='threads', type=int, default=1, help='number of threads (TODO)')


    args = parser.parse_args()
    return args






##########################################################################
######################          DEPRECIATED     ##########################
##########################################################################
##########################################################################

################### TOKENIZATION WITH JUMAN (doesnt work) #################
def tokenize_juman(line, tokenizer):
    def get_token(morph):
        return morph.repname if morph.repname != "" else morph.genkei
    result = tokenizer.analysis(line.decode('utf8'))
    return ' '.join(get_token(morph) for morph in result.mrph_list())

def tokenize_file_juman(file):
    tokenizer = Jumanpp()
    file = open(file)
    return '\n'.join(tokenize(l, tokenizer) for l in file)

############# TOKENIZATION WITH RAKUTENMA (prohibitively slow) ############
def tokenize_rakuten(line, tokenizer):
    return u' '.join(guess_stem(x)[0] for [x, pos] in tokenizer.tokenize(line.decode('utf8')))

def tokenize_file_rakuten(file):
    model = 'rakuten_model_ja.min.json'
    tokenizer = RakutenMA(json.loads(open(model).read()))
    tokenizer.hash_func = RakutenMA.create_hash_func(tokenizer, 15)
    return '\n'.join(tokenize_rakuten(l, tokenizer) for l in open(file))

##########################################################################
##########################################################################
##########################################################################
##########################################################################

def wc(f):
    return int(commands.getstatusoutput('wc -l %s' % f)[1].split()[0])


def tokenize(line, pos=False):
    def get_tok(morph):
        morph = morph.split()
        if not morph or morph[-1] == 'NIL' or len(morph) < 4:
            return ''
        if pos:
            return '%s:%s' % (morph[2], morph[3])
        return morph[2]

    def clean(text):
        text = text.replace("'", '').replace('"', '')
        return text

    cmd = "echo '%s' | juman" % clean(line)
    juman_out = commands.getstatusoutput(cmd)[1]
    return ' '.join(get_tok(tok) for tok in juman_out.split('\n') if get_tok(tok))


def tokenize_file(f, pos=False):
    return '\n'.join(tokenize(line, pos) for line in tqdm(open(f), total=wc(f)))





def main(args):
    # tokenize input
    out = tokenize_file(args.corpus, args.pos)

    # write to output
    with open(args.output, 'w') as f:
        f.write(out)

    # write vocab if specified
    if args.vocab is not None:
        vocab = Counter(out.split())
        vocab = '\n'.join(v for v in vocab.keys())
        with open(args.vocab, 'w') as f:
            f.write(vocab)



if __name__ == '__main__':
    args = process_command_line()
    main(args)






