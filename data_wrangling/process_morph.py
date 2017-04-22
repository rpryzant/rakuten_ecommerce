"""
python process_morph.py ../data/bpe_small/total.outputs ../data/raw_morph/ total.inputs.morph total.outputs

python process_morph.py [output file] [morph root dir]
                        e.g. bpe_small/total.outputs
"""
import sys
import os
import re

def mostly_english(token):
    """ tests whether a token has some alpha chars in it
    """
    return sum([1 if c.isalpha() else 0 for c in token]) / float(len(token)) > 0.2


def can_open(path):
    """ since os.path wants escaped semicolons, just try opening the file
    """
    try:
        with open(path) as f:
            if len(next(f)) > 0:
                return True
    except:
        return False


def find_morph_file(morph_root, shop, category, item_id):
    """ finds the right morphology file for an item
    """
    morph_category_dir = category + '_desc_pro'
    #../data/raw_morph/bag_desc_pro/bag.desc.izm-izm\:10000736.juman.pre 
    item_number = re.findall('\d+', item_id)[0]

    file = '%s.desc.%s.juman.pre' % (category, item_id)
    file_path = os.path.join(morph_root, morph_category_dir, file)

    if can_open(file_path):
        return file_path

    return None


def extract_text(path):
    """ extracts text from a tab-sep morpho file
    """
    out = ''
    for line in open(path):
        try:
            token = line.strip().split('\t')[1].replace(' ', '')
            if not token.isdigit() and not token.lower() == 'EOF' and not mostly_english(token):
                out += ' ' + token
        except:
            continue
    return out

outputs = open(sys.argv[1])
morph_root = sys.argv[2]

with open(sys.argv[3], 'a') as out_inputs:
    with open(sys.argv[4], 'a') as out_outputs:
        for output in outputs:
            # get labels
            [sales, shop, price, category, item_id] = output.strip().split('|')

            # get corresponding morpho file
            morph_file = find_morph_file(morph_root, shop, category, item_id)
            if not morph_file:
                continue

            # pull out text from that file                
            item_text = extract_text(morph_file)
            if not item_text:
                continue

            # write to output                
            out_outputs.write(output.strip() + '\n')
            out_inputs.write(item_text + '\n')







