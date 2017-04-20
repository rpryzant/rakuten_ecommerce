import time
import sys
import numpy as np


PAD = 0
UNK = 1

class SmallConfig:
    batch_size = 128
    max_len = 400

    embedding_size = 16
    hidden_size = 16
    num_layers = 1
    num_attention_units = 16
    prediction_hidden_size = 16
    optimizer = 'Adam'
    gradient_clip = 5.0

    learning_rate = 0.0001    # [1.0 for sgd, 0.0003 for adam] work well
    dropout_rate = 0.4

    num_epochs = 14

    attention_keys = 'word_vectors'        # [rnn_states, word_vectors]
    attention_type = 'bahdanau'         # [bahdanau, dot, fc]



class MediumConfig:
    batch_size = 128
    max_len = 400

    embedding_size = 32
    hidden_size = 32
    num_layers = 1
    num_attention_units = 32
    prediction_hidden_size = 32
    optimizer = 'Adam'
    gradient_clip = 5.0

    learning_rate = 0.0001    # [1.0 for sgd, 0.0003 for adam] work well
    dropout_rate = 0.4

    num_epochs = 14

    attention_keys = 'word_vectors'        # [rnn_states, word_vectors]
    attention_type = 'bahdanau'         # [bahdanau, dot, fc]


class LargeConfig:
    batch_size = 128
    max_len = 400

    embedding_size = 64
    hidden_size = 64
    num_layers = 1
    num_attention_units = 64
    prediction_hidden_size = 64
    optimizer = 'Adam'
    gradient_clip = 5.0

    learning_rate = 0.0001    # [1.0 for sgd, 0.0003 for adam] work well
    dropout_rate = 0.4

    num_epochs = 14

    attention_keys = 'word_vectors'        # [rnn_states, word_vectors]
    attention_type = 'bahdanau'         # [bahdanau, dot, fc]








class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)
