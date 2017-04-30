"""
python trainer.py test ../data_wrangling/total.inputs.bpe ../data_wrangling/total.outputs ../data_wrangling/bpe.vocab

python trainer.py test ../data/example_data/bag.inputs.bpe ../data/example_data/bag.outputs ../data/example_data/bpe.vocab
"""

import cPickle
import models
import utils
import input_pipeline
import tensorflow as tf
import os
import argparse # option parsing





def main(args):
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    c = utils.make_config(args)


    d = input_pipeline.DataInputPipeline(
            args.inputs,
            args.vocab,
            args.labels,
            c)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # Or whichever device you would like to use
#    gpu_options = tf.GPUOptions(allow_growth=True)
    sess =  tf.Session()#config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    m = models.Model(c, sess, d)

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


    if args.output is not None:
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


        print '\nINFO: dumping output to ', args.output
        utils.write_pickle(source_out, args.output)

    print '\n\nINFO:  \(^o^)/   done    \(^o^)/'








if __name__ == '__main__':
    args = utils.process_command_line()
    main(args)




