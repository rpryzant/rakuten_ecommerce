"""

python eval_words.py  /Users/rapigan/Desktop/words /Users/rapigan/Desktop/datasets/bpe/health /Users/rapigan/Desktop/datasets/morph/health


python batch_eval.py [words root] [bpe data root] [morph data root]
"""
import argparse # option parsing
import evaluator
import os
import commands
import time
from tqdm import tqdm
import re
import numpy as np

TEST_IDS = '../john_code/%s.multi_candid.all3'


def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description='Usage') # add description
    # positional arguments
    parser.add_argument('root', metavar='root', type=str, help='root for words')
    parser.add_argument('bpe_data', metavar='bpe_data', type=str, help='bpe data root')
    parser.add_argument('morph_data', metavar='morph_data', type=str, help='morph data root')


    args = parser.parse_args()
    return args


def parse_tables(eval_output):
    raw_tables = re.split('={3,}[\w ()/]+={3,}', eval_output)[1:]
    header = []
    values = []

    for i, raw_table in enumerate(raw_tables):
        table = np.array(zip(*[iter(raw_table.replace(' ', '_').split())]*4))

        table_type = 'BASE' if i == 0 else 'BPE' if i == 1 else 'MORPH'
        for r in range(table.shape[0])[1:]:
            for c in range(table.shape[1])[1:]:
                header.append( '%s-%s-%s' % (table_type, table[0, c], table[r, 0]))
                values.append(table[r, c])

    return ','.join([h for h in header]), ','.join([e for e in values])





def gen_evaluations(args):
    bpe_inputs = os.path.join(args.bpe_data, 'inputs')
    morph_inputs = os.path.join(args.morph_data, 'inputs')
    outputs = os.path.join(args.bpe_data, 'outputs')


    for dataset in ['health']:
        for template in ['/rnn/%s/ja/best_%s_flipped', '/rnn/%s/ja/best_%s_notflipped']:
            bpe_best = args.root +  template % (dataset, 'bpe')
            morph_best = args.root + template % (dataset, 'morph')
            cmd = 'python evaluator.py %s %s %s %s %s %s' % (outputs, bpe_inputs, morph_inputs, TEST_IDS % dataset, bpe_best, morph_best)
            print cmd
            out = commands.getstatusoutput(cmd)[1]
            best_table_header, best_table_values = parse_tables(out)
            print best_table_header
            print '%s,%s' % (dataset, 'False' if 'notflip' in template else 'True'), best_table_values





def main(args):
    gen_evaluations(args)




if __name__ == '__main__':
    args = process_command_line()
    main(args)





