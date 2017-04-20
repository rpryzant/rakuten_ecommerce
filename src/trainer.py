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

    # optional arguments
#    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default=None, help='model directory')
 
    args = parser.parse_args()
    return args




def main(model_path):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    c = utils.Config()
    d = input_pipeline.DataInputPipeline(
        '../data/example_data/bag.inputs.bpe',
        '../data/example_data/bpe.vocab',
        '../data/example_data/bag.outputs',
        c)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Or whichever device you would like to use
#    gpu_options = tf.GPUOptions(allow_growth=True)
    sess =  tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    m = model.Model(c, sess, d)

    sess.run(tf.global_variables_initializer())    


    print 'INFO: starting training...'
    prog = utils.Progbar(target=d.get_num_batches())
    epoch = 1
    while(True):
        epoch_loss = 0
        for i, batch in enumerate(d.batch_iter()):
            sales_hat, price_hat, shop_hat, category_hat, loss = \
                m.train_on_batch(*batch)
            prog.update(i+1, [('train loss', loss)])
            epoch_loss += loss
        print '\n INFO: EPOCH ', epoch, ' MEAN LOSS: ', epoch_loss / float(d.get_num_batches())
        print 'INFO: saving checkpoint...'
        m.save(model_path)
        print 'INFO: ...done!'
        epoch += 1
if __name__ == '__main__':
    args = process_command_line()
    main(args.checkpoint)




