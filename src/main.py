import model
import utils
import input_pipeline
import tensorflow as tf
import os


def main():
    c = utils.Config()
    d = input_pipeline.DataInputPipeline(
        '../data/example_data/bag.inputs.bpe',
        '../data/example_data/bpe.vocab',
        '../data/example_data/bag.outputs',
        c)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Or whichever device you would like to use
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess =  tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

    m = model.Model(c, sess, d)

    sess.run(tf.global_variables_initializer())    

#    (example, lens, sales, price, shop, category)  = next(d.batch_iter())

    while(True):
        for (example, lens, sales, price, shop, category) in d.batch_iter():
            print m.train_on_batch(example, lens, sales, price, shop, category)



if __name__ == '__main__':
    main()