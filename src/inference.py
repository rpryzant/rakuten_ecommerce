"""
python inference.py -o rnn_states-bahdanau-large/out.pkl -t bahdanau -k rnn_states, -s large -g 1 rnn_states-bahdanau-large/ ../data_wrangling/total.inputs.bpe ../data_wrangling/total.outputs ../data_wrangling/bpe.vocab

"""

import cPickle
import models
import utils
import input_pipeline
import tensorflow as tf
import os
import argparse # option parsing




def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1)




def main(args):
    c = utils.make_config(args)
    d = input_pipeline.DataInputPipeline(
            args.inputs,
            args.vocab,
            args.labels,
            c)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#    gpu_options = tf.GPUOptions(allow_growth=True)
    sess =  tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    m = models.Model(c, sess, d)

    print 'INFO: loading model from checkpoint...'
    m.load(dir=args.checkpoint)
    print 'INFO: done!'

    sess.run(tf.global_variables_initializer())    

    print 'INFO: starting inference...'
    prog = utils.Progbar(target=d.get_num_batches())

    source_out = {'source': [], 'len': [], 'attn': [], 'ids': []}
    sales_out = {'label': [], 'pred': []}
    price_out = {'label': [], 'pred': []}
    shop_out = {'label': [], 'pred': []}
    category_out = {'label': [], 'pred': []}
    loss_out = []

    for i, batch in enumerate(d.batch_iter()):
        sales_hat, price_hat, shop_hat, category_hat, loss, attn = \
            m.test_on_batch(*batch[:-1]) 

        prog.update(i, [('train loss', loss)])

        # record results
        source, source_len, log_sales, price, shop, category, ids = batch

        source_out['source'] += source
        source_out['len'] += source_len
        source_out['attn'] += attn.tolist()
        source_out['ids'] += ids

        sales_out['label'] += log_sales
        sales_out['pred'] += sales_hat.tolist()

        price_out['label'] += price
        price_out['pred'] += price_hat.tolist()

        shop_out['label'] += shop
        shop_out['pred'] += shop_hat.tolist()

        category_out['label'] += category
        category_out['pred'] += category_hat.tolist()

        loss_out += loss

    if args.output is not None:
        print '\nINFO: dumping output to ', args.output
        write_pickle(source_out, args.output)

    print 'INFO:  done \(^o^)/'



if __name__ == '__main__':
    args = utils.process_command_line()
    main(args)


