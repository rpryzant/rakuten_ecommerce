import utils
import numpy as np
import math

class DataInputPipeline(object):
    def __init__(self, text, vocab, labels, config):
        self.batch_size = config.batch_size
        self.max_len = config.max_len

        self.word_to_index = self.parse_vocab(vocab)
        self.index_to_word = {i: x for x, i in self.word_to_index.iteritems()}

        # todo - refactor
        self.broken, self.sales, self.shop, self.price, \
            self.category, self.ids, self.examples, self.lens, \
            self.shop_to_index, self.category_to_index = \
                self.prepare_data(labels, text)

        self.index_to_shop = {i: x for x, i in self.shop_to_index.iteritems()}
        self.index_to_category = {i: x for x , i in self.category_to_index.iteritems()}

        self.N = len(self.examples)

        self.indices = np.arange(self.N)


    def get_vocab_size(self):
        return len(self.word_to_index)

    def get_num_shops(self):
        return len(self.shop_to_index)

    def get_num_batches(self):
        return self.N / self.batch_size


    def get_num_categories(self):
        return len(self.category_to_index)

    def batch_iter(self):
        i = 0
        for _ in range(self.get_num_batches()):
            example = self.examples[i: i+self.batch_size]
            lens = self.lens[i: i+self.batch_size]
            sales = self.sales[i: i+self.batch_size]
            price = self.price[i: i+self.batch_size]
            shop = self.shop[i: i+self.batch_size]
            category = self.category[i: i+self.batch_size]
            ids = self.ids[i: i + self.batch_size]
            yield example, lens, sales, price, shop, category, ids

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
        parse = filter(lambda x: len(x) > 2, parse)

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
        out_ids = []
        for i, ([log_sales, shop_name, price, category, item_id], line) in \
            enumerate(zip(parse, open(text))):

            try:
                out_ids.append(item_id)
                out_sales.append(float(log_sales))
                out_shop.append(shop_name_mapping[shop_name])
                out_price.append(math.log(float(price)))
                out_category.append(category_mapping[category])

                example = [self.word_to_index.get(tok, utils.UNK) for tok in line.split()]
                example = post_pad(example)# [::-1]    # reverse for reverse
                out_examples.append(example)
                out_lens.append(np.count_nonzero(example))
            except:
                broken.append(i)
        return broken, out_sales, out_shop, out_price, out_category, out_ids, \
                out_examples, out_lens, shop_name_mapping, category_mapping


    def reconstruct(self, seq, shop_name=None, category=None):
        text, l = seq
        # reverse for reverse
        return [self.index_to_word.get(x, 'UNK') for x in text], \
                self.index_to_shop.get(shop_name, None), \
                self.index_to_category.get(category, None)


    def parse_vocab(self, f):
        d = {}
        for i, l in enumerate(open(f)):
            d[l.split()[0]] = i + 2      # +2 to reserve 0 for pad, 1 for unk
        d['UNK'] = 1
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
