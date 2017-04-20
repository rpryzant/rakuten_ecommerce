"""
python process_data.py ../data/bag_desc/ ../data/bag_sales.txt
"""
import pandas as pd
import sys
import os
import math
import re


data_dir = sys.argv[1]

# read in categories
dir_contents = os.listdir(data_dir)
categories = set([x.split('_')[0] for x in dir_contents \
                    if x.split('_')[0] + '_sales.txt' in dir_contents])

# extract inputs (title, description 1 & 2) and outputs (log sales, shop name, price, category)
for category in categories:
    print 'INFO: working on category ', category

    desc_dir = os.path.join(data_dir, category + '_desc')
    sales_filepath = os.path.join(data_dir, category + '_sales.txt')
    desc_type = os.path.basename(os.path.normpath(desc_dir)).replace('_', '.') + '.'

    inputs = ''
    outputs = ''

    # parse each item in the category
    sales = pd.read_csv(sales_filepath, sep='\t')
    skipped = 0
    for item_id, sales in zip(sales.item_id, sales.unit_sales):
        description_file = os.path.join(desc_dir, desc_type + item_id)
        description = open(description_file).read().split('\t')
        if len(description) != 11: 
            skipped += 1
            continue
        [item_title, item_id, price, description1, description2, _, img_url, num_reviews, avg_rating, shop_name, category_id] = \
            description
        # todo - title and description as seperate inputs?
        item_description = re.sub('\s+', ' ', item_title + ' ' + description1 + ' ' + description2).strip()

        inputs += item_description + '\n'
        outputs += '%s|%s|%s|%s\n' % (str(math.log(float(sales + 0.00001))), shop_name, price, category)


    with open(category + '.inputs', 'w') as f:
        f.write(inputs)
    with open(category + '.outputs', 'w') as f:
        f.write(outputs)

# concatenate inputs
print 'INFO: joining categories...'
cmd = 'cat ' + ' '.join(x + '.inputs' for x in categories) + ' > total.inputs'
os.system(cmd)

cmd = 'cat ' + ' '.join(x + '.outputs' for x in categories) + ' > total.outputs'
os.system(cmd)

# tokenize inputs
print 'INFO: tokenizing inputs...'
cmd = './tokenize_inputs.sh total.inputs'

print 'Done!'





