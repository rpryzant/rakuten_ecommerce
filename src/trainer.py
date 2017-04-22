"""
python trainer.py test ../data_wrangling/total.inputs.bpe ../data_wrangling/total.outputs ../data_wrangling/bpe.vocab
"""


import model
import utils
import input_pipeline
import tensorflow as tf
import os
import argparse # option parsing





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




