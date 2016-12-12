# -*- coding: utf-8 -*-
from sketch_tables import *
from heap import MinHeap
from hash_strategy import *
from update_strategy import *
from lossy_update_strategy import *

import math

from itertools import izip


DEFAULT_N = 10


class CountMinSketch(object):
    """
    A minimalist implementation of count-min sketches
    """
    def __init__(self, delta, epsilon, depth=None, width=None,
                 table_class=ListBackedSketchTable,
                 hash_strategy=NaiveHashingStrategy,
                 update_strategy=NaiveUpdateStrategy,
                 lossy_strategy=NoLossyUpdateStrategy):

        if depth is None:
            depth = int(math.ceil(math.log(1.0 / delta)))

        if width is None:
            width = int(math.ceil(math.e / epsilon))

        self.table = table_class(depth, width)

        # iffy work-around to support receiving both a class and a pre-initialized strategy
        if isinstance(hash_strategy, type):
            self.hash = hash_strategy(depth, width)
        else:
            self.hash = hash_strategy

        if isinstance(update_strategy, type):
            self.updater = update_strategy()
        else:
            self.updater = update_strategy

        if isinstance(lossy_strategy, type):
            self.lossy = lossy_strategy()

        else:
            self.lossy = lossy_strategy

    def insert(self, item, count=1):
        hashes = self.hash(item)
        self.lossy(self.table)
        return self.updater(self.table, hashes, count)

    def update(self, items, counts=None):
        if counts is None:
            [self.insert(item) for item in items]

        else:
            [self.insert(item, count) for item, count in izip(items, counts)]

    def get(self, item):
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
    def __init__(self, delta, epsilon, depth=None, width=None, n=DEFAULT_N,
                 table_class=ListBackedSketchTable,
                 hash_strategy=NaiveHashingStrategy,
                 update_strategy=NaiveUpdateStrategy,
                 lossy_strategy=NoLossyUpdateStrategy):
        super(TopNCountMinSketch, self).__init__(delta, epsilon, depth, width,
                                                 table_class=table_class,
                                                 hash_strategy=hash_strategy,
                                                 update_strategy=update_strategy,
                                                 lossy_strategy=lossy_strategy)
        self._init_top_n(n)

    def _init_top_n(self, n):
        self.n = n
        self.heap = MinHeap()
        self.heap_full = False
        self.top_n = {}

    def insert(self, item, count=1):
        new_min_count = super(TopNCountMinSketch, self).insert(item, count)
        self._update_top_n(item, new_min_count)
        return new_min_count

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

    def most_common(self, k=None):
        """
        Return the most common k items, for k <= n.
        Accepting a parameter to conform to python's Counter interface
        :param k: The number of top items to return, at most n, as this function was initialized
        :return: The top k items observed by this CM sketch, and how often they were counter
        """
        if k is None or k > self.n:
            k = self.n

        return [item[::-1] for item in sorted(self.top_n.values(), reverse=True)[:k]]

    def __str__(self):
        return str(self.most_common())
