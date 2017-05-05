"""
python batch_eval.py ../src/BPE_CHOCO/ ../src/MORPH_CHOCO/ ../data/large/bpe/choco/ ../data/large/morph/with_pos/choco/ choco

python batch_eval.py [bpe out root] [morph out root] [bpe data root] [morph data root] [type]
"""
import argparse # option parsing
import evaluator
import os
import commands
import time
from tqdm import tqdm

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


def parse_table(eval_output):
    raw_tables = re.split('={3,}[\w ()/]+={3,}', eval_output)[1:]
    header = []
    values = []

    for i in range(len(x)):
        table = np.array(zip(*[iter(raw_tables[i].replace(' ', '_').split())]*4))

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

    for run in tqdm(os.listdir(args.bpe_out)):
        bpe_run_dir = os.path.join(args.bpe_out, run)
        bpe_best = os.path.join(bpe_run_dir, '%s-best-%s' % (args.type, run))
        bpe_worst = os.path.join(bpe_run_dir, '%s-worst-%s' % (args.type, run))

        morph_run_dir = os.path.join(args.morph_out, run)
        morph_best = os.path.join(morph_run_dir, '%s-best-%s' % (args.type, run))
        morph_worst = os.path.join(morph_run_dir, '%s-worst-%s' % (args.type, run))

        start = time.time()
        cmd = 'python evaluator.py %s %s %s %s %s %s' % (outputs, bpe_inputs, morph_inputs, TEST_IDS, bpe_best, morph_best)
        print cmd
        out = commands.getstatusoutput(cmd)[1]
        table_header, table_values = parse_evaluator_output(out)

        yield run.replace('-', ','), table_header, table_values


def main(args):
    run_info = ','.join(['product_type', 'word_type', 'wv_size', 'reversed', 'attn_key', 
                        'attn_type', 'attn_order', 'mixing_ratio', 
                        'hidden_size', 'attn_units', 'pred_units'])

    with open('OUT', 'a') as out:
        for i, run, header, values in enumerate(gen_evaluations(args)):
            if i == 0:
                out.write(run_info + ',' + header + '\n')
            out.write(run + ',' + values + '\n')




if __name__ == '__main__':
    args = process_command_line()
    main(args)





