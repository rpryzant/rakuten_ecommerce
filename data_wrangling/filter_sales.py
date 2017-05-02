"""
filters data and applies categorical labelings to the 'sales' attribute

items with sales in the top 30% are given 1, bottom 30% 0. otherwise discarded

python filter_sales.py ../data/large/bpe/health/

"""

import sys
import numpy as np
from bisect import bisect_right

ZERO_SALES = -11.512925465   # sigh

def get_sales(row):
    return float(row.split('|')[0])


def get_top_bottom(fp):
    """ choose some examples (ignoring most of the 0 sales) and return the top/bottom 30%
    """
    with open(fp) as f:
        sales = sorted([(get_sales(r), i) for (i, r) in enumerate(f)], reverse=True)
        m = min([x[0] for x in sales])
        min_index = next(i for i, x in enumerate(sales) if x[0] == m)
        selected_examples = sales[:min_index + 500]
        c = int(len(selected_examples) * 0.3)
        top = [x[1] for x in selected_examples[:c]]
        bottom = [x[1] for x in selected_examples[-c:]]
        return top, bottom

def relabel(row, l):
    """ replaces log(sales) in a row with label "l"
    """
    parts = row.split('|')
    parts = [str(l)]  + parts[1:]
    return '|'.join(x for x in parts)


data_dir = sys.argv[1]

inputs_path = data_dir + '/inputs'
outputs_path = data_dir + '/outputs'


top, bottom = get_top_bottom(outputs_path)

with open(data_dir + '/inputs.binary', 'a') as input_out:
    with open(data_dir + '/outputs.binary', 'a') as output_out:
        with open(inputs_path) as inputs:
            with open(outputs_path) as outputs:

                for i, (in_row, out_row) in enumerate(zip(inputs, outputs)):
                    if i in top:
                        input_out.write(in_row)
                        output_out.write(relabel(out_row, 1))
                    elif i in bottom:
                        input_out.write(in_row)
                        output_out.write(relabel(out_row, 0))
                        
