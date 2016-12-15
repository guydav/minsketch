# -*- coding: utf-8 -*-
"""Two implementations of ideas from "Less Hashing, Same Performance: Building
a Better Bloom Filter," by Kirsch and Mitzenmacher (2008):
https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf
"""
import math
from gmpy2 import next_prime

import count_min_sketch
import hash_strategy
import lossy_strategy
import sketch_tables
import update_strategy


class HashPairCMSketch(count_min_sketch.TopNCountMinSketch):
    r"""A single hash-pair count-min sketch, based on Kirsch & Mitzenmacher
    (2008). This essentially ignores the epsilon parameter, and provides a
    robust implementation of a count-min sketch using only two hash functions.
    """

    def __init__(self,
                 delta,
                 epsilon,
                 hash_gen=None,
                 n=count_min_sketch.DEFAULT_N,
                 table_class=sketch_tables.ListBackedSketchTable,
                 update_strategy=update_strategy.NaiveUpdateStrategy,
                 lossy_strategy=lossy_strategy.NoLossyUpdateStrategy):
        """Most parameters conform to TopNCountMinSketch, except for the delta
        and hash_gen:
        :param delta: How often (on expectation) we accept an error. Ignored
        here, but received to conform to the same initialization signature of
        the rest of the count-min sketches.
        :param epsilon: What margin (on expectation) we accept for an error
        :param hash_gen: We allow receiving a hash generator to be able to
        share them between different instances, if we want to guarantee their
        independence.
        :param n: The top-N items to track.
        :param table_class: The class of table to use in backing this CM-sketch
        :param update_strategy: The updating strategy to use (currently naive
        or convservative)
        :param lossy_strategy: The lossy window strategy to use (naive, no
        threshold, threshold of 1, threshold of window count, threshold of the
        square root of the window count)

        """
        width = next_prime(int(math.ceil(2 * math.e / epsilon)))
        depth = int(
            math.ceil(math.log(1.0 / (epsilon - epsilon / (2 * math.e**2)))))
        print('The effective delta is {delta}'.format(delta=math.exp(-1 *
                                                                     depth)))

        if hash_gen is None:
            hash_gen = hash_strategy.UniversalHashFunctionGenerator(
                hash_strategy.ARBITRARY_LARGE_PRIME_NUMBER)

        super(HashPairCMSketch, self).__init__(
            delta, epsilon, depth, width, n, table_class,
            hash_strategy.DoubleHashingStrategy(depth, width, hash_gen),
            update_strategy, lossy_strategy)


class MultiHashPairTopNCMSketch(count_min_sketch.TopNCountMinSketch):
    r"""From the previously referenced paper:
    "Given such a result, it is straightforward to obtain a variation that uses
    :math:`2\frac{\ln \frac{1}{\delta}}{\ln \frac{1}{\epsilon}}` pairwise
    independent hash functions and achieves the desired failure probability
    :math:`\delta`: simply build
    :math:`2\frac{\ln \frac{1}{\delta}}{\ln \frac{1}{\epsilon}}`
    independent copies of this data structure, and always answer a point query
    with the minimum estimate given by one of those copies."

    I believe the second 2 is a mistake - so I only build
    :math:`\frac{\ln \frac{1}{\delta}}{\ln \frac{1}{\epsilon}}` rounded up
    sketches.

    """

    def __init__(self,
                 delta,
                 epsilon,
                 n=count_min_sketch.DEFAULT_N,
                 table_class=sketch_tables.ListBackedSketchTable,
                 update_strategy=update_strategy.NaiveUpdateStrategy,
                 lossy_strategy=lossy_strategy.NoLossyUpdateStrategy):

        self._init_top_n(n)
        self.count = int(
            math.ceil(math.log(1.0 / delta) / math.log(1.0 / epsilon)))

        hash_gen = hash_strategy.UniversalHashFunctionGenerator(
            hash_strategy.ARBITRARY_LARGE_PRIME_NUMBER)

        self.sketches = [HashPairCMSketch(
            delta,
            epsilon,
            hash_gen,
            table_class=table_class,
            update_strategy=update_strategy,
            lossy_strategy=lossy_strategy) for _ in range(self.count)]

    def insert(self, item, count=1):
        new_min_count = min(
            [sketch.insert(item, count) for sketch in self.sketches])
        self._update_top_n(item, new_min_count)
        return new_min_count

    def get(self, item):
        return min([sketch.get(item) for sketch in self.sketches])
