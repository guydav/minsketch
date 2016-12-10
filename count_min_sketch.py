import math

from sketch_tables import *
from heap import MinHeap

import random
from itertools import izip


ARBITRARY_LARGE_PRIME_NUMBER = 4294967291  # 280829369862134719390036617067
DEFAULT_N = 10


class UniversalHashFunctionGenerator(object):
    """
    Implementation of a universal hash function family generator, as described in
    Cormen et al.'s Introduction to Algorithms.
    """
    def __init__(self, m):
        self.a_set = {None, }
        self.b_set = {None, }
        self.m = m

    def __call__(self, *args, **kwargs):
        a = None
        b = None

        while a in self.a_set and b in self.b_set:
            a = random.randint(1, ARBITRARY_LARGE_PRIME_NUMBER - 1)
            b = random.randint(0, ARBITRARY_LARGE_PRIME_NUMBER - 1)

        self.a_set.add(a)
        self.b_set.add(b)

        return lambda x: ((a * x + b) % ARBITRARY_LARGE_PRIME_NUMBER) % self.m


class CountMinSketch(object):
    """
    A minimalist implementation of count-min sketches
    """
    def __init__(self, epsilon, delta, table_class=ListBackedSketchTable):
        depth = int(math.ceil(math.log(1.0 / delta)))
        width = int(math.ceil(math.e / epsilon))
        self.table = table_class(depth, width)
        hash_gen = UniversalHashFunctionGenerator(width)
        self.hashes = [hash_gen() for _ in range(depth)]

    def hash(self, item):
        if not isinstance(item, int) or isinstance(item, long):
            item = hash(item)

        return [self.hashes[i](item) for i in range(self.table.depth)]

    def insert(self, item, count=1):
        hashes = self.hash(item)
        return min([self.table.increment(i, hashes[i], count) for i in range(self.table.depth)])

    def batch_insert(self, items, counts=None):
        if counts is None:
            [self.insert(item) for item in items]

        else:
            [self.insert(item, count) for item, count in izip(items, counts)]

    def point_query(self, item):
        hashes = self.hash(item)
        return min([self.table.get(i, hashes[i]) for i in range(self.table.depth)])

    def inner_product_query(self, first_item, second_item):
        first_hashes = self.hash(first_item)
        second_hashes = self.hash(second_item)

        return min([self.table.get(i, first_hashes[i]) * self.table.get(i, second_hashes[i])
                    for i in range(self.table.depth)])


class TopNCountMinSketch(CountMinSketch):
    """
    Implementation of a CountMinSketch supporting tracking the top-N items
    """
    def __init__(self, epsilon, delta, n=DEFAULT_N, table_class=ListBackedSketchTable):
        super(TopNCountMinSketch, self).__init__(epsilon, delta, table_class)
        self.n = n
        self.heap = MinHeap()
        self.heap_full = False
        self.top_n = {}

    def insert(self, item, count=1):
        new_min_count = super(TopNCountMinSketch, self).insert(item, count)
        self._update_top_n(item, new_min_count)

    def _update_top_n(self, item, new_min_count):
        if not self.heap_full or new_min_count > self.heap.peek()[0]:
            if item in self.top_n:
                existing_pair = self.top_n[item]
                existing_pair[0] = new_min_count
                self.heap.heapify()

            else:
                pair = [new_min_count, item]
                self.top_n[item] = pair

                if self.heap_full:
                    old_pair = self.heap.push_pop(pair)
                    del self.top_n[old_pair[1]]

                else:
                    self.heap.push(pair)
                    if len(self.heap) == self.n:
                        self.heap_full = True

    def most_common(self):
        return [item[::-1] for item in sorted(self.top_n.values(), reverse=True)]

    def __str__(self):
        return str(self.most_common())


class HashPairCountMinSketch(CountMinSketch):
    """
    A single hash-pair count-min sketch, based on Kirsch & Mitzenmacher (2007)
    """
    def __init__(self, epsilon, delta, hashes=None, table_class=ListBackedSketchTable):
        # TODO: for the Mitzenmacher scheme, the width needs to be prime, and constructed differently?
        # width = prime > 2epsilon / e
        #
        super(HashPairCountMinSketch, self).__init__(epsilon, delta, table_class)
        hash_gen = UniversalHashFunctionGenerator(self.table.width)

        if hashes:
            self.first_hash, self.second_hash = hashes

        else:
            self.first_hash = hash_gen()
            self.second_hash = hash_gen()

    def hash(self, item):
        if not isinstance(item, int) or isinstance(item, long):
            item = hash(item)

        first = self.first_hash(item)
        second = self.second_hash(item)
        return [first + i * second for i in range(self.table.depth)]






def test():
    count = 100000
    numbers = [random.randint(1, ARBITRARY_LARGE_PRIME_NUMBER - 1) for _ in xrange(count)]
    counts = [random.randint(10, 1000) for _ in xrange(count)]

    cms = TopNCountMinSketch(0.005, 10 ** -7)

    for (number, count) in izip(numbers, counts):
        cms.insert(number, count)

    print(cms.table.depth, cms.table.width, cms.table.depth * cms.table.width)

    total_error = 0
    percent_error = 0

    for i in range(count):
        error = float(cms.point_query(numbers[i]) - counts[i])
        total_error += error
        percent_error = error / counts[i]

    print('Total error:', total_error / sum(counts))
    print('Average percent error:', percent_error / count)
    print(cms)


def text_test():
    import re
    from collections import Counter
    from functools import partial
    from timeit import timeit
    from asizeof import asizeof

    words = re.findall(r'\w+', open('hamlet.txt').read().lower())
    print('{length} words in the document'.format(length=len(words)))

    epsilon = 0.001
    delta = 10 ** -5
    list_backed = TopNCountMinSketch(epsilon, delta, table_class=ListBackedSketchTable)
    print list_backed.table.depth, list_backed.table.width
    array_backed = TopNCountMinSketch(epsilon, delta, table_class=ArrayBackedSketchTable)
    matrix_backed = TopNCountMinSketch(epsilon, delta, table_class=NumpyMatrixBackedSketchTable)
    bitarray_backed = TopNCountMinSketch(epsilon, delta, table_class=BitarrayBackedSketchTable)
    counter = Counter()

    print('Runtimes:')
    print(timeit(partial(list_backed.batch_insert, words), number=1))
    print(timeit(partial(array_backed.batch_insert, words), number=1))
    print(timeit(partial(matrix_backed.batch_insert, words), number=1))
    print(timeit(partial(bitarray_backed.batch_insert, words), number=1))
    print(timeit(partial(counter.update, words), number=1))
    print('')

    print('Approximated memory footprints:')
    print(asizeof(list_backed))
    print(asizeof(array_backed))
    print(asizeof(matrix_backed))
    print(asizeof(bitarray_backed))
    print(asizeof(counter))
    print('')

    print('Top-10s:')
    print(list_backed)
    print(array_backed)
    print(matrix_backed)
    print(bitarray_backed)
    print(counter.most_common(10))
    print('')

    # print('% differences:')
    # for cms_entry, counter_entry in izip(cms.most_common(), counter.most_common(10)):
    #     if cms_entry[0] != counter_entry[0]:
    #         print('Word mismatch:', cms_entry[0], counter_entry[0])
    #     else:
    #         diff = float(cms_entry[1] - counter_entry[1]) / counter_entry[1]
    #         print('{word}: {percent:.4f}'.format(word=cms_entry[0], percent=diff))


def text_test_bitarrays():
    import re
    from collections import Counter
    from functools import partial
    from timeit import timeit
    from asizeof import asizeof

    words = re.findall(r'\w+', open('test/shakespeare.txt').read().lower())
    print('{length} words in the document'.format(length=len(words)))

    epsilon = 0.001
    delta = 10 ** -5
    list_backed = TopNCountMinSketch(epsilon, delta, table_class=ListBackedSketchTable)
    print list_backed.table.depth, list_backed.table.width
    struct_backed = TopNCountMinSketch(epsilon, delta, table_class=BitarrayBackedSketchTableWithStruct)
    bitarray_backed = TopNCountMinSketch(epsilon, delta, table_class=BitarrayBackedSketchTable)
    counter = Counter()

    print('Runtimes:')
    print(timeit(partial(list_backed.batch_insert, words), number=1))
    print(timeit(partial(struct_backed.batch_insert, words), number=1))
    print(timeit(partial(bitarray_backed.batch_insert, words), number=1))
    print(timeit(partial(counter.update, words), number=1))
    print('')

    print('Approximated memory footprints:')
    print(asizeof(list_backed))
    print(asizeof(struct_backed))
    print(asizeof(bitarray_backed))
    print(asizeof(counter))
    print('')

    print('Top-10s:')
    print(list_backed)
    print(struct_backed)
    print(bitarray_backed)
    print(counter.most_common(10))
    print('')


if __name__ == '__main__':
    text_test_bitarrays()
