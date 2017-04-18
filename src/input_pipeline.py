import utils
import numpy as np

class DataInputPipeline(object):
    def __init__(self, text, vocab, labels, config):
        self.batch_size = config.batch_size
        self.max_len = config.max_len

        self.word_to_index = self.parse_vocab(vocab)
        self.index_to_word = {i: x for x, i in self.word_to_index.iteritems()}

        # todo - refactor
        self.broken, self.sales, self.shop, self.price, \
            self.category, self.examples, self.lens, \
            self.shop_to_index, self.category_to_index = \
                self.prepare_data(labels, text)

        self.index_to_shop = {i: x for x, i in self.shop_to_index.iteritems()}
        self.index_to_category = {i: x for x , i in self.category_to_index.iteritems()}

        self.N = len(self.examples)

        self.indices = np.arange(self.N)


    def get_vocab_size(self):
        return len(self.word_to_index)


    def batch_iter(self):
        i = 0
        while i + self.batch_size < self.N:
            example = self.examples[i: i+self.batch_size]
            lens = self.lens[i: i+self.batch_size]
            sales = self.sales[i: i+self.batch_size]
            price = self.price[i: i+self.batch_size]
            shop = self.shop[i: i+self.batch_size]
            category = self.category[i: i+self.batch_size]

            yield example, lens, sales, price, shop, category

            i += self.batch_size



    def prepare_data(self, f, text):
        # TODO - REFACTOR

        def post_pad(y, pad=utils.PAD):
            new = [pad] * self.max_len
            new[:len(y)] = y
            return new[:self.max_len]


        parse = []
        for i, l in enumerate(open(f)):
            parse.append(l.strip().split('|'))

        shop_names = set(x[1] for x in parse)
        categories = set(x[3] for x in parse)

        shop_name_mapping = {x: i for i, x in enumerate(shop_names)}
        category_mapping = {x: i for i, x in enumerate(categories)}

        broken = []
        out_sales = []
        out_shop = []
        out_price = []
        out_category = []
        out_examples = []
        out_lens = []
        for i, ([log_sales, shop_name, price, category], line) in enumerate(zip(parse, open(text))):
            try:
                out_sales.append(float(log_sales))
                out_shop.append(shop_name_mapping[shop_name])
                out_price.append(float(price))
                out_category.append(category_mapping[category])

                example = [self.word_to_index.get(tok, utils.UNK) for tok in line.split()]
                example = post_pad(example)[::-1]
                out_examples.append(example)
                out_lens.append(np.count_nonzero(example))
            except:
                broken.append(i)
        return broken, out_sales, out_shop, out_price, out_category, \
                out_examples, out_lens, shop_name_mapping, category_mapping


    def reconstruct(self, seq, shop_name=None, category=None):
        text, l = seq
        return [self.index_to_word.get(x, 'UNK') for x in text][::-1], \
                self.index_to_shop.get(shop_name, None), \
                self.index_to_category.get(category, None)


    def parse_vocab(self, f):
        d = {}
        for i, l in enumerate(open(f)):
            d[l.split()[0]] = i + 2      # +1 to reserve 0 for pad, 1 for unk
        d[1] = 'UNK'
        return d



if __name__=='__main__':
    c = utils.Config()
    d = DataInputPipeline(
        '../data/example_data/bag.inputs.bpe',
        '../data/example_data/bpe.vocab',
        '../data/example_data/bag.outputs',
        c)
    sources, targets = next(d.batch_iter())
    r = d.reconstruct(sources[0], targets[0][1], targets[0][3])
    print r[1], r[2]
    print ' '.join(x for x in r[0])
    print targets[0]