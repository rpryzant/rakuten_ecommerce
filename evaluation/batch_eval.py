"""
python batch_eval.py ../src/BPE_CHOCO/ ../src/MORPH_CHOCO/ ../data/large/bpe/choco/ ../data/large/morph/with_pos/choco/ choco

python batch_eval.py ../src/BPE_CHOCO2/ ../src/MORPH_CHOCO2/ /Users/rapigan/Desktop/datasets/bpe/choco /Users/rapigan/Desktop/datasets/morph/choco choco



python batch_eval.py [bpe out root] [morph out root] [bpe data root] [morph data root] [type]
"""
import argparse # option parsing
import evaluator
import os
import commands
import time
from tqdm import tqdm
import re
import numpy as np

TEST_IDS = '../john_code/choco.multi_candid.all'


def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description='Usage') # add description
    # positional arguments
    parser.add_argument('bpe_out', metavar='bpe_out', type=str, help='root for model outputs')
    parser.add_argument('morph_out', metavar='morph_out', type=str, help='root for model outputs')
    parser.add_argument('bpe_data', metavar='bpe_data', type=str, help='bpe data root')
    parser.add_argument('morph_data', metavar='morph_data', type=str, help='morph data root')
    parser.add_argument('type', metavar='type', type=str, help='product type')

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

    for run in os.listdir(args.bpe_out):
        bpe_run_dir = os.path.join(args.bpe_out, run)
        bpe_best = os.path.join(bpe_run_dir, '%s-best-%s' % (args.type, run))
        bpe_worst = os.path.join(bpe_run_dir, '%s-worst-%s' % (args.type, run))


        morph_run_dir = os.path.join(args.morph_out, run)
        morph_best = os.path.join(morph_run_dir, '%s-best-%s' % (args.type, run))
        morph_worst = os.path.join(morph_run_dir, '%s-worst-%s' % (args.type, run))

        cmd = 'python evaluator.py %s %s %s %s %s %s' % (outputs, bpe_inputs, morph_inputs, TEST_IDS, bpe_best, morph_best)
        out = commands.getstatusoutput(cmd)[1]
        best_table_header, best_table_values = parse_tables(out)


        yield run.replace('-', ','), best_table_header, best_table_values


def main(args):
    run_info = ','.join(['product_type', 'word_type', 'wv_size', 'reversed', 'attn_key', 
                        'attn_type', 'attn_order', 'mixing_ratio', 
                        'hidden_size', 'attn_units', 'pred_units'])

    for i, (run, header, values) in enumerate(gen_evaluations(args)):
        if i == 0:
            print run_info + ',' + header
        print run + ',' +  values




if __name__ == '__main__':
    args = process_command_line()
    main(args)





