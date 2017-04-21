"""
python trainer.py test ../data_wrangling/total.inputs.bpe ../data_wrangling/total.outputs ../data_wrangling/bpe.vocab
"""


import model
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
    parser.add_argument('-t', '--attention-type', dest='attention_type', type=str, default=None, 
        help='what kind of attention to use: [bahdanau, dot, fc]')
    parser.add_argument('-k', '--attention-keys', dest='attention_keys', type=str, default=None, 
        help='what kind of attention keys to use: [rnn_states, word_vectors]')
    parser.add_argument('-s', '--model-size', dest='model_size', type=str, default='medium', 
        help='what size model: [small, medium, large]')
    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='0', 
        help='which gpu to run on')


#    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default=None, help='model directory')
    args = parser.parse_args()
    return args



def make_config(args):
    if args.model_size == 'small':
        c = utils.SmallConfig()
    elif args.model_size == 'medium':
        c = utils.MediumConfig()
    elif args.model_size == 'large':
        c = utils.LargeConfig()

    if args.attention_type is not None:
        c.attention_type = args.attention_type
    if args.attention_keys is not None:
        c.attention_keys = args.attention_keys

    return c


def main(args):
    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    c = utils.make_config(args)


    d = input_pipeline.DataInputPipeline(
            args.inputs,
            args.vocab,
            args.labels,
            c)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # Or whichever device you would like to use
#    gpu_options = tf.GPUOptions(allow_growth=True)
    sess =  tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    m = model.Model(c, sess, d)

    sess.run(tf.global_variables_initializer())    


    print 'INFO: starting training...'
    prog = utils.Progbar(target=d.get_num_batches())
    epoch = 1
    for _ in range(c.num_epochs):
        epoch_loss = 0
        for i, batch in enumerate(d.batch_iter()):
            sales_hat, price_hat, shop_hat, category_hat, loss = \
                m.train_on_batch(*batch[:-1])
            prog.update(i+1, [('train loss', loss)])
            epoch_loss += loss
        print '\n INFO: EPOCH ', epoch, ' MEAN LOSS: ', epoch_loss / float(d.get_num_batches())
        print 'INFO: saving checkpoint...'
        m.save(args.checkpoint)
        print 'INFO: ...done!'
        epoch += 1
if __name__ == '__main__':
    args = utils.process_command_line()
    main(args)




