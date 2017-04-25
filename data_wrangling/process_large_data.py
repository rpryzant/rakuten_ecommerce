"""
process a large data dump from john: 
    files have one item per line,
    but there are no sales units that I know of

python process_large_data.py ../data/large/raw/health.test.out.tr

"""
import sys
import os
import re


f = sys.argv[1]

description = open(f)

inputs = ''
outputs = ''
skipped = 0
for line in description:
    split_line = line.split('\t')
    if len(split_line) != 11: 
        skipped += 1
        continue

    [item_title, item_id, price, description1, description2, _, img_url, num_reviews, avg_rating, shop_name, category_id] = \
        split_line

    item_description = re.sub('\s+', ' ', item_title + ' ' + description1 + ' ' + description2).strip()

    inputs += item_description  + '\n'

    outputs += 'TODO|%s|%s|CATEGORY_OMITTED|%s\n' % (shop_name, price, item_id)


with open(os.path.basename(f) + '.inputs', 'w') as out:
    out.write(inputs)
with open(os.path.basename(f) + '.outputs', 'w') as out:
    out.write(outputs)

