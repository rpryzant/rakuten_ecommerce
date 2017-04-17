import utils
import numpy as np

class DataInputPipeline(object):
    def __init__(self, text, vocab, labels, config):
        self.batch_size = config.batch_size
        self.max_len = config.max_len

        self.word_to_index = self.parse_vocab(vocab)
        self.source = self.prepare_source(text)
        self.targets, self.shop_name_mapping, self.category_mapping = \
            self.prepare_targets(labels)

        len(self.targets)
        len(self.source)
        self.N = len(self.source)

        self.indices = np.arange(self.N)


    def batch_iter(self):
        i = 0
        while i + self.batch_size < self.N:
            yield self.source[i: i+self.batch_size], self.targets[i: i + self.batch_size]
            i += self.batch_size



    def prepare_targets(self, f):
        parse = []
        for i, l in enumerate(open(f)):
            parse.append(l.strip().split('|'))

        shop_names = set(x[1] for x in parse)
        categories = set(x[3] for x in parse)

        shop_name_mapping = {x: i for i, x in enumerate(shop_names)}
        category_mapping = {x: i for i, x in enumerate(categories)}

        out = []
        for [log_sales, shop_name, price, category] in parse:
            out.append([float(log_sales), 
                        shop_name_mapping[shop_name], 
                        float(price), 
                        category_mapping[category]])

        return out, shop_name_mapping, category_mapping



    def prepare_source(self, f):
        def post_pad(y, pad=utils.PAD):
            new = [pad] * self.max_len
            new[:len(y)] = y
            return new

        out = []
        for i, l in enumerate(open(f)):
            text = [self.word_to_index.get(tok, utils.UNK) for tok in l.split()]
            out.append(post_pad(text)[::-1])   # reverse inputs

        return out


    def parse_vocab(self, f):
        d = {}
        for i, l in enumerate(open(f)):
            d[l.split()[0]] = i
        return d



if __name__=='__main__':
    c = utils.Config()
    d = DataInputPipeline(
        '../data/example_data/bag.inputs.bpe',
        '../data/example_data/bpe.vocab',
        '../data/example_data/bag.outputs',
        c)
    print next(d.batch_iter())

