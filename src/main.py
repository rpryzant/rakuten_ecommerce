import model
import utils
import input_pipeline
import tensorflow as tf



def main():
    c = utils.Config()
    d = input_pipeline.DataInputPipeline(
        '../data/example_data/bag.inputs.bpe',
        '../data/example_data/bpe.vocab',
        '../data/example_data/bag.outputs',
        c)
    sess = tf.Session()

    m = model.Model(c, sess, d)

    sess.run(tf.global_variables_initializer())    

    (example, lens, sales, price, shop, category)  = next(d.batch_iter())

    while(True):
        print m.train_on_batch(example, lens, sales)



if __name__ == '__main__':
    main()