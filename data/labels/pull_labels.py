"""
python pull_labels.py [label file] [source file] [target file] [vocab]

python pull_labels.py choco_multi_candid3.txt ../../data_wrangling/total.inputs.bpe ../../data_wrangling/total.outputs

"""

import sys
import sys; sys.path.append('../../')    # sigh 
import src.input_pipeline

label_ids = [l.strip().split()[1] for l in open(sys.argv[1])]


source = open(sys.argv[2])
targets = open(sys.argv[3])

source_out = ''
targets_out = ''

for src_l, target_l in zip(source, targets):
    if target_l.strip().split('|')[-1] in label_ids:
        source_out += src_l
        targets_out += target_l

print source_out
