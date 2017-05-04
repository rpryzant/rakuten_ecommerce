import time
import sys
import numpy as np
import argparse # option parsing
import cPickle


PAD = 0
UNK = 1







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
    parser.add_argument('-t', '--attention-type', dest='attention_type', type=str, default='fc', 
        help='what kind of attention to use: [bahdanau, dot, fc]')
    parser.add_argument('-k', '--attention-keys', dest='attention_keys', type=str, default='rnn_states', 
        help='what kind of attention keys to use: [rnn_states, word_vectors]')
    parser.add_argument('-g', '--gpu', dest='gpu', type=str, default='0', 
        help='which gpu to run on')    
    parser.add_argument('-ao', '--attention-order', dest='attention_order', type=str, default='before_split', 
        help='when to build attention graph: [before_split, after_split]')    
    parser.add_argument('-o', '--output', dest='output', type=str, default=None, 
        help='dump output to file')
    parser.add_argument('-mr', '--mixing-ratio', dest='mixing_ratio', type=float, default=0.5, 
        help='mixing ratio between sales loss and other loss')
    parser.add_argument('-rg', '--reverse-gradients', dest='reverse_gradients', type=bool, default=True, 
        help='whether to flip gradients from non-sales prediction heads')
    parser.add_argument('-es', '--embedding-size', dest='embedding_size', type=int, default=32, 
        help='size of word embeddings')
    parser.add_argument('-hs', '--hidden-size', dest='hidden_size', type=int, default=32, 
        help='size of hidden representations')
    parser.add_argument('-au', '--attention-units', dest='attention_units', type=int, default=32, 
        help='num attentional units')
    parser.add_argument('-ph', '--prediction-units', dest='prediction_units', type=int, default=32, 
        help='num prediction hidden units')
    parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=12,
        help='num training epochs')
    parser.add_argument('-train', '--train', action='store_true', 
        help='run training (will otherwise attempt to load saved checkpoint)')
    
    args = parser.parse_args()
    return args





def write_pickle(o, fname):
    with open(fname, 'w') as f:
        cPickle.dump(o, f, -1)




def make_config(args):
    print '=' * 80
    print args
    print '=' * 80

    c = Config()

    c.num_epochs = args.epochs
    c.prediction_hidden_size = args.prediction_units
    c.attention_units = args.attention_units
    c.hidden_size = args.hidden_size
    c.embedding_size = args.embedding_size
    c.reverse_gradients = args.reverse_gradients
    c.mixing_ratio = args.mixing_ratio
    c.attention_order = args.attention_order
    c.attention_type = args.attention_type
    c.attention_keys = args.attention_keys

    return c



class Config:
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
    attention_order = 'before_split'    # [before_split, after_split]
    reverse_gradients = True
    mixing_ratio = 0.5         # how much of loss comes from sales







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
