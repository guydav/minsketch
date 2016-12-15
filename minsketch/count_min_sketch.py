# -*- coding: utf-8 -*-
"""This module implements the basic count-min sketch class, from which most
others inherit. It also offers an implementation of a count-min sketch that
keeps track of some number of most commonly recurring items.
"""

import math
from itertools import izip

import hash_strategy
import heap
import lossy_strategy
import sketch_tables
import update_strategy


DEFAULT_N = 100


class CountMinSketch(object):
    """A minimalist implementation of count-min sketches
    """

    def __init__(self,
                 delta,
                 epsilon,
                 depth=None,
                 width=None,
                 table_class=sketch_tables.ListBackedSketchTable,
                 hash_strategy=hash_strategy.NaiveHashingStrategy,
                 update_strategy=update_strategy.NaiveUpdateStrategy,
                 lossy_strategy=lossy_strategy.NoLossyUpdateStrategy):
        """Delta and epsilon define our error threshold. They allow specifying
        how often an error is acceptable within a certain range of the true
        count.

        It then allocates a table of the given type with the calculated or
        provided width and depth, and uses the optional strategy choices to
        implement regular or double hashing, naive or conservative update, and
        naive or lossy counting.
        :param delta: How often (on expectation) we accept an error
        :param epsilon: What margin (on expectation) we accept for an error
        :param depth: Provide a value to override the default depth calculation
        :param width: Provide a value to override the default width calculation
        :param table_class: The class of table to use in backing this CM-sketch
        :param hash_strategy: The hashing strategy to use (currently regular or
        double)
        :param update_strategy: The updating strategy to use (currently naive
        or convservative)
        :param lossy_strategy: The lossy window strategy to use (naive, no
        threshold, threshold of 1, threshold of window count, threshold of the
        square root of the window count)
        """

        if depth is None:
            depth = int(math.ceil(math.log(1.0 / delta)))

        if width is None:
            width = int(math.ceil(math.e / epsilon))

        self.table = table_class(depth, width)

        # iffy work-around to support receiving both a class and a
        # pre-initialized strategy
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
        """Insert an item with a given count into the CM-sketch,
        using the provided hashing, lossy counting, and updating strategies

        :param item: The item to insert. Must be a number or sensibly hashable
        (say, a string)
        :param count: The count to insert with, defaults to 1
        :return: The new minimal value of this item in the CM-sketch.
        """
        hashes = self.hash(item)
        self.lossy(self.table)
        return self.updater(self.table, hashes, count)

    def update(self, items, counts=None):
        """Implement a group update of items and counts

        :param items: The items to update in
        :param counts: The count of each item. If exists, assumed to be the
        same size as items.
        :return: None
        """
        if counts is None:
            [self.insert(item) for item in items]

        else:
            [self.insert(item, count) for item, count in izip(items, counts)]

    def get(self, item):
        """Retrieve the current minimum value associated with an item

        :param item: The item to retrieve
        :return: The best current estimate for how often it appeared so far
        """
        hashes = self.hash(item)
        return min(
            [self.table.get(i, hashes[i]) for i in range(self.table.depth)])

    def inner_product_query(self, first_item, second_item):
        """Not expanded upon, but a simple implementation of the idea of an
        inner-product query using a CM-sketch, as dicussed in the original
        paper

        :param first_item: The first item queried
        :param second_item: The second item queried
        :return: The best estimate for their inner-product within the CM-sketch
        """
        first_hashes = self.hash(first_item)
        second_hashes = self.hash(second_item)

        return min([self.table.get(i, first_hashes[i]) *
                    self.table.get(i, second_hashes[i])
                    for i in range(self.table.depth)])


class TopNCountMinSketch(CountMinSketch):
    """Implementation of a CountMinSketch supporting tracking the top-N items.

    Inspired in part by the following blog post:
    https://tech.shareaholic.com/2012/12/03/the-count-min-sketch-how-to-count-over-large-keyspaces-when-about-right-is-good-enough/
    """

    def __init__(self,
                 delta,
                 epsilon,
                 depth=None,
                 width=None,
                 n=DEFAULT_N,
                 table_class=sketch_tables.ListBackedSketchTable,
                 hash_strategy=hash_strategy.NaiveHashingStrategy,
                 update_strategy=update_strategy.NaiveUpdateStrategy,
                 lossy_strategy=lossy_strategy.NoLossyUpdateStrategy):
        super(TopNCountMinSketch, self).__init__(
            delta,
            epsilon,
            depth,
            width,
            table_class=table_class,
            hash_strategy=hash_strategy,
            update_strategy=update_strategy,
            lossy_strategy=lossy_strategy)
        self._init_top_n(n)

    def _init_top_n(self, n):
        """Initialize the framework required to track the top-N items.
        Extracted to a separate method to ease subclassing.
        :param n: The number of top items to track
        :return: None
        """
        self.n = n
        self.heap = heap.MinHeap()
        self.heap_full = False
        self.top_n = {}

    def insert(self, item, count=1):
        """Insert the item, and update the top-N registry.

        :param item: The item to insert. Must be a number or sensibly hashable
        (say, a string)
        :param count: The count to insert with, defaults to 1
        :return: The new minimal count for this item
        """
        new_min_count = super(TopNCountMinSketch, self).insert(item, count)
        self._update_top_n(item, new_min_count)
        return new_min_count

    def _update_top_n(self, item, new_min_count):
        """Update the top-N registry for a given item.
        The post-condition maintains that the N most recurring items so far
        will be in the top_n dictionary and heap, and that no more than N items
        will be in either.

        :param item: The item just inserted.
        :param new_min_count: Its new minimal count
        :return: None
        """
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
        """Return the most common k items, for k <= n.
        Accepting a parameter to conform to python's Counter interface, and
        returning the results in their style
        :param k: The number of top items to return, at most n, as this
        function was initialized
        :return: The top k items observed by this CM sketch, and how often they
        were seen
        """
        if k is None or k > self.n:
            k = self.n

        return [item[::-1]
                for item in sorted(
                    self.top_n.values(), reverse=True)[:k]]

    def __str__(self):
        return str(self.most_common())
