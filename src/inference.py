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
    parser.add_argument('-d', '--dump-attention', action='store_true', help='dump attentional scores and figures')
 
    args = parser.parse_args()
    return args




def main(model_path):
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

    print 'INFO: loading model from checkpoint...'
    m.load(dir=model_path)
    print 'INFO: done!'

    sess.run(tf.global_variables_initializer())    

    print 'INFO: starting inference...'
    prog = utils.Progbar(target=d.get_num_batches())

    source_out = {'source': [], 'attn': []}
    sales_out = {'label': [], 'pred': []}
    price_out = {'label': [], 'pred': []}
    shop_out = {'label': [], 'pred': []}
    category_out = {'label': [], 'pred': []}
    loss_out = []

    for i, batch in enumerate(d.batch_iter()):
        sales_hat, price_hat, shop_hat, category_hat, loss, attn = \
            m.test_on_batch(*batch)
        prog.update(i, [('train loss', loss)])

        # record results
        source, source_len, log_sales, price, shop, category = batch

        source_out['source'] += source
        source_out['attn'] += attn.tolist()

        sales_out['label'] += log_sales
        sales_out['pred'] += sales_hat.tolist()

        price_out['label'] += price
        price_out['pred'] += price_hat.tolist()

        shop_out['label'] += shop
        shop_out['pred'] += shop_hat.tolist()

        category_out['label'] += category
        category_out['pred'] += category_hat.tolist()

        loss_out += loss

        if i > 4:
            break


    print source_out['source'][0], source_out['attn'][0]

if __name__ == '__main__':
    args = process_command_line()
    main(args.checkpoint)


