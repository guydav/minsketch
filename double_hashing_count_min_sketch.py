# -*- coding: utf-8 -*-
from count_min_sketch import *
from sketch_tables import *
from hashing_strategy import DoubleHashingStrategy

import math
from gmpy2 import next_prime


class HashPairCMSketch(TopNCountMinSketch):
    """
    A single hash-pair count-min sketch, based on Kirsch & Mitzenmacher (2008):
    https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf

    "In particular, for w ≥ 2e/\epsilon and \delta ≥ ln 1/\epsilon(1 − 1/2e2)"
    => My assumption is that they mean d, rather than delta, no?
    """
    def __init__(self, epsilon, hash_gen=None, n=DEFAULT_N, table_class=ListBackedSketchTable):
        self._init_top_n(n)
        width = next_prime(int(math.ceil(2 * math.e / epsilon)))
        # TODO: check if they mean d or delta here
        depth = int(math.ceil(math.log(1.0 / (epsilon - epsilon / (2 * math.e ** 2)))))
        # TODO: If they did, compute and report effective delta (presumably e^(-depth))

        self.table = table_class(depth, width)
        self.hash = DoubleHashingStrategy(depth, width, hash_gen)


# TODO: Iron out the inconsistency - 2 * fraction copies of the data strcutre require 2 * 2 * fraction
# TODO: hash functions - not just 2x. Presumably they meant construct (fraction) instances?


class MultiHashPairTopNCMSketch(TopNCountMinSketch):
    """
    From the previously referenced paper:
    "Given such a result, it is straightforward to obtain a variation that uses 2(ln 1/\delta)/(ln 1/\epsilon)
    pairwise independent hash functions and achieves the desired failure probability \delta: simply
    build 2(ln 1/\delta)/(ln 1/\epsilon)independent copies of this data structure, and always answer a
    point query with the minimum estimate given by one of those copies."
    """
    def __init__(self, epsilon, delta, n=DEFAULT_N, table_class=ListBackedSketchTable):
        self._init_top_n(n)
        self.count = int(math.ceil(math.log(1.0 / delta) / math.log(1.0 / epsilon)))
        self.sketches = [HashPairCMSketch(epsilon, table_class=table_class) for _ in range(self.count)]

        # The loop is used to share the same hash generator between them, guaranteeing independence
        for _ in range(self.count - 1):
            self.sketches.append(HashPairCMSketch(epsilon, hash_gen=self.sketches[0].hash.hash_gen,
                                                  table_class=table_class))

    def insert(self, item, count=1):
        new_min_count = min([sketch.insert(item, count) for sketch in self.sketches])
        self._update_top_n(item, new_min_count)

