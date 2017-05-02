"""
filters data and applies categorical labelings to the 'sales' attribute

items with sales in the top 30% are given 1, bottom 30% 0. otherwise discarded

python filter_sales.py


"""

import sys
import numpy as np



def get_sales(row):
    return float(row.split('|')[0])


def sales_cutoffs(fp):
    with open(fp) as f:
        sales = [get_sales(r) for (i, r) in enumerate(f)]        
        print sum(1 if x < -10 else 0 for x in sales) * 1.0 / len(sales)
#        print np.mean([x[0] for x in sales])
        sales = sorted(sales)
        cutoff = int(len(sales) * 0.3)
        top = [x for x in sales[-cutoff:]]
#        print top
#        print cutoff



inputs = sys.argv[1]
outputs = sys.argv[2]



sales_cutoffs(outputs)


