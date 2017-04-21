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




def process_command_line():
    """
    Return a 1-tuple: (args list).
    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    """
    parser = argparse.ArgumentParser(description='Usage') # add description
    # positional arguments
    parser.add_argument('checkpoint', metavar='checkpoint', type=str, help='model directory')
    parser.add_argument('inputs', metavar='inputs', type=str, help='model inputs')
    parser.add_argument('labels', metavar='labels', type=str, help='model outputs (labels)')
    parser.add_argument('vocab', metavar='vocab', type=str, help='vocabulary')

    # optional arguments
    parser.add_argument('-t', '--attention-type', dest='attention_type', type=str, default='fc', 
        help='what kind of attention to use: [bahdanau, dot, fc]')
    parser.add_argument('-k', '--attention-keys', dest='attention_keys', type=str, default='rnn_states', 
        help='what kind of attention keys to use: [rnn_states, word_vectors]')
    parser.add_argument('-s', '--model-size', dest='model_size', type=str, default='medium', 
        help='what size model: [small, medium, large]')
    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='0', 
        help='which gpu to run on')    
    parser.add_argument('-o', '--output', dest='output', type=str, default=None, help='dump output to file')
 
    args = parser.parse_args()
    return args



def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1)




def make_config(args):
#    checkpoint_dir = os.path.basename(os.path.normpath(checkpoint_dir))
#    [attention_keys, attention_type] = checkpoint_dir.split('-')
    print args
    if args.model_size == 'small':
        c = utils.SmallConfig()
    elif args.model_size == 'medium':
        c = utils.MediumConfig()
    elif args.model_size == 'large':
        c = utils.LargeConfig()

    c.attention_type = args.attention_type
    c.attention_keys = args.attention_keys


    return c



def main(args):
    c = make_config(args)
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
    args = process_command_line()
    main(args)


