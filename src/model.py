
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.rnn.python.ops import rnn
from graph_module import GraphModule
from collections import namedtuple
import numpy as np

class Model:

    def __init__(self, config, sess, dataset):
        self.sess = sess

        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.vocab_size = dataset.get_vocab_size()
        self.num_shops = dataset.get_num_shops()
        self.num_categories = dataset.get_num_categories()

        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_attention_units = config.num_attention_units
        self.prediction_hidden_size = config.prediction_hidden_size
        self.optimizer = config.optimizer
        self.gradient_clip = config.gradient_clip
        self.train_dropout = config.dropout_rate


        # training
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # inputs
        self.source = tf.placeholder(tf.int32, [self.batch_size, self.max_len], name='source')
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name='source_len')

        # outputs
        self.log_sales = tf.placeholder(tf.float32, [self.batch_size], name='log_sales')
        self.shop = tf.placeholder(tf.int32, [self.batch_size], name='shop')
        self.price = tf.placeholder(tf.float32, [self.batch_size], name='price')
        self.category = tf.placeholder(tf.int32, [self.batch_size], name='category')

        source_encoding = self.encode(self.source, self.source_len)

        with tf.variable_scope('sales'):
            sales_hat, sales_loss = self.regressor(self.log_sales,
                                                   source_encoding, 
                                                   reverse_grads=False, 
                                                   name='sales')
        with tf.variable_scope('price'):
            price_hat, price_loss = self.regressor(self.price, 
                                                   source_encoding, 
                                                   reverse_grads=True, 
                                                   name='price')
        with tf.variable_scope('shop'):
            shop_logits, shop_loss = self.classifier(self.shop, 
                                                     source_encoding, 
                                                     num_classes=self.num_shops,
                                                     reverse_grads=True, 
                                                     name='shop')
        with tf.variable_scope('category'):
            category_logits, category_loss = self.classifier(self.category, 
                                                             source_encoding, 
                                                             num_classes=self.num_categories,
                                                             reverse_grads=True, 
                                                             name='category')



        self.loss = sales_loss + price_loss + shop_loss + category_loss
        self.sales_hat = sales_hat
        self.price_hat = price_hat
        self.shop_hat = shop_logits
        self.category_hat = category_logits
        self.train_step = self.optimize(self.loss)


    def train_on_batch(self, source, source_len, log_sales, price, shop, category, learning_rate=0.0003):
        _, sales_hat, price_hat, shop_hat, category_hat, loss = self.sess.run([self.train_step, self.sales_hat, self.price_hat, self.shop_hat, self.category_hat, self.loss],
                                feed_dict={
                                    self.source: source,
                                    self.source_len: source_len,
                                    self.log_sales: log_sales,
                                    self.price: price,
                                    self.shop: shop,
                                    self.category: category,
                                    self.learning_rate: learning_rate,
                                    self.dropout: self.train_dropout
                                })
        return sales_hat.tolist(), log_sales, price_hat.tolist(), price, np.argmax(shop_hat, axis=1).tolist(), shop, np.argmax(category_hat, axis=1).tolist(), category, loss





    def optimize(self, loss):
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            clip_gradients=self.gradient_clip,
            optimizer=self.optimizer,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
        return train_op





    def regressor(self, labels, encoder_output, reverse_grads=False, name='regressor'):
        encoder_output_output = encoder_output.outputs
        encoder_output_output_shape = encoder_output_output.get_shape()

        encoder_output_att_values = encoder_output.attention_values
        encoder_output_att_values_shape = encoder_output_att_values.get_shape()

        encoder_att_values_length = encoder_output.attention_values_length

        if reverse_grads:
            encoder_output_output = reverse_grad(encoder_output_output)
            encoder_output_output.set_shape(encoder_output_output_shape)

            encoder_output_att_values = reverse_grad(encoder_output_att_values)
            encoder_output_att_values.set_shape(encoder_output_att_values_shape)

        scores, attentional_context = self.run_attention(encoder_output_output,
                                                         encoder_output_att_values,
                                                         encoder_att_values_length)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=attentional_context,
            num_outputs=self.prediction_hidden_size,
            activation_fn=tf.nn.tanh,
            scope='%s_fc' % name)  

        preds = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=1,
            activation_fn=None,
            scope='%s_pred' % name)
        preds = tf.squeeze(preds)


        loss = tf.nn.l2_loss(preds - labels)
        loss = loss / self.batch_size  # mean per-example loss
#        loss = tf.reduce_mean(loss)

        return preds, loss


    def classifier(self, labels, encoder_output, num_classes, reverse_grads=False, name='classifier'):
        encoder_output_output = encoder_output.outputs
        encoder_output_output_shape = encoder_output_output.get_shape()

        encoder_output_att_values = encoder_output.attention_values
        encoder_output_att_values_shape = encoder_output_att_values.get_shape()

        encoder_att_values_length = encoder_output.attention_values_length

        if reverse_grads:
            encoder_output_output = reverse_grad(encoder_output_output)
            encoder_output_output.set_shape(encoder_output_output_shape)

            encoder_output_att_values = reverse_grad(encoder_output_att_values)
            encoder_output_att_values.set_shape(encoder_output_att_values_shape)

        scores, attentional_context = self.run_attention(encoder_output_output,
                                                         encoder_output_att_values,
                                                         encoder_att_values_length)

        fc1 = tf.contrib.layers.fully_connected(
            inputs=attentional_context,
            num_outputs=self.prediction_hidden_size,
            activation_fn=tf.nn.tanh,
            scope='%s_fc' % name)  

        logits = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=num_classes,
            activation_fn=None,
            scope='%s_pred' % name)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

        mean_loss = tf.reduce_mean(losses)

        return logits, mean_loss








    def encode(self, source, source_len):
        with tf.variable_scope('embedding'):
            source_embedded = self.get_embeddings(self.source)
        with tf.variable_scope('encoder'):
            encoder_output = self.run_encoder(source_embedded, self.source_len)
        return encoder_output

    def get_embeddings(self, source):
        E = tf.get_variable('E',
                            shape=[self.vocab_size, self.embedding_size])
        embedding = tf.nn.embedding_lookup(E, source)
        return embedding


    def run_encoder(self, source, source_len):
        cell = self.build_rnn_cell()
        encoder = StackedBidirectionalEncoder(cell)
#        encoder = BidirectionalEncoder(cell)
        encoder_output = encoder(source, source_len)
        return encoder_output





    def run_attention(self, encoder_output_output,
                            encoder_output_att_values,
                            encoder_att_values_length):
        attention_fn = AttentionLayerDot(num_units=self.num_attention_units)
        normalized_scores, attention_context = attention_fn(
            query=tf.zeros_like(encoder_output_output[:, 0, :]),
            keys=encoder_output_output,
            values=encoder_output_att_values,
            values_length=encoder_att_values_length)

        return normalized_scores, attention_context



    def build_rnn_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             input_keep_prob=(1 - self.dropout))
        stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers,
                                                   state_is_tuple=True)
        return stacked_cell
#        return cell






def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)





EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class StackedBidirectionalEncoder(GraphModule):
    def __init__(self, cell, name='bidirectional_encoder'):
        super(StackedBidirectionalEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs, final_fw_state, final_bw_state = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=self.cell._cells,
            cells_bw=self.cell._cells,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)

        # Concatenate outputs and states of the forward and backward RNNs
#        outputs = tf.concat(2, outputs)
        final_state = final_fw_state, final_bw_state

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=lengths)




class BidirectionalEncoder(GraphModule):
    def __init__(self, cell, name='default_bidirectional'):
        super(BidirectionalEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs_pre, final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell,
            cell_bw=self.cell,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)
        # Concatenate outputs and states of the forward and backward RNNs
        outputs = tf.concat(2, outputs_pre)

        return outputs, final_state








class AttentionLayer(GraphModule):
    def __init__(self, num_units=128, name='attention'):
        GraphModule.__init__(self, name)
        self.num_units = num_units

    def score_fn(self, keys, query):
        """Computes the attention score"""
        raise NotImplementedError

    def _build(self, query, keys, values, values_length):
        values_depth = values.get_shape().as_list()[-1]

        att_keys = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=self.num_units,
            activation_fn=None,
            scope="att_keys")

        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=self.num_units,
            activation_fn=None,
            scope="att_query")

        scores = self.score_fn(att_keys, att_query)


        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)


        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, values_depth])


        return (scores_normalized, context)



class AttentionLayerDot(AttentionLayer):
    def score_fn(self, keys, query):
        """Calculates a batch- and timweise dot product"""
        return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])






