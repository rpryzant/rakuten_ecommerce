"""
appends sales data to output files with that field missing

python append_sales.py ../data/large/sales_labels/choco_all_sales.txt ../data/large/bpe/choco/choco.model_outputs > ../data/large/bpe/choco/choco.model_outputs2

"""


import sys
import math

f = open(sys.argv[1])
next(f)

sales = {x.split('\t')[0]: str(math.log(float(x.split('\t')[1]) + 0.00001)) for x in f}

model_outputs = open(sys.argv[2])

out = ''
for row in model_outputs:
    item_id = row.split('|')[-1].strip()
    print row.replace('TODO', sales[item_id]).strip()
