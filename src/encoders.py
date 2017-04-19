import tensorflow as tf
from collections import namedtuple
from graph_module import GraphModule
from tensorflow.contrib.rnn.python.ops import rnn



EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class StackedBidirectionalEncoder(GraphModule):
    """ multi-layer bidirectional encoders
    """
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

        # Concatenate states of the forward and backward RNNs
        final_state = final_fw_state, final_bw_state

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=lengths)


class BidirectionalEncoder(GraphModule):
    """ single-layer bidirectional encoder
    """
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

        # Concatenate outputs of the forward and backward RNNs
        outputs = tf.concat(2, outputs_pre)

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=lengths)

