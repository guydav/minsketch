# -*- coding: utf-8 -*-
"""This model implements count-mean-min sketches, as described by Goyal and
Daumé (2011):
http://www.aclweb.org/anthology/D12-1100

The design here is a little iffy - there's a fair amount of code duplication
between the two classes. Perhaps a mixin would fit better, or changing the
abstractions.
"""
from itertools import izip

from numpy import median

import count_min_sketch
import double_hashing


# TODO: Re-consider design here at some point - Mixins? Different abstraction?


class CountMeanMinSketch(count_min_sketch.TopNCountMinSketch):
    """A count-mean-min sketch, based on the regular TopNCountMinSketch.

    The two main changes include using error-estimates from the updater
    (regular or conservative), and returning min between the minimum of the
    regularly queried values, and the median of the values after accounting
    for the error estimates.
    """

    def get(self, item):
        hashes = self.hash(item)
        values = [self.table.get(i, hashes[i])
                  for i in range(self.table.depth)]
        error_estimates = self.updater.count_mean_error_estimates(self.table,
                                                                  values)
        return min(
            median(
                sorted([value - error
                        for value, error in izip(values, error_estimates)])),
            min(values))

    def most_common(self, k=None):
        """Re-query the top-n keys using the count-mean query algorithm, and
        then return the top k of the re-queried results
        :param k: The number of top items to return, at most n, as this
        function was initialized
        :return: The top k items observed by this CM sketch, and how often they
        were seen
        """
        mean_values = [(key, self.get(key)) for _, key in self.top_n]
        mean_values.sort(key=lambda pair: pair[1], reverse=True)

        if k is None or k > self.n:
            k = self.n

        return mean_values[:k]


class HashPairCountMeanMinSketch(double_hashing.HashPairCMSketch):
    """A count-mean-min sketch, based on the HashPair CM SKETCH.

        The two main changes include using error-estimates from the updater
        (regular or conservative), and returning min between the minimum of
        the regularly queried values, and the median of the values after
        accounting for the error estimates.
        """

    def get(self, item):
        hashes = self.hash(item)
        values = [self.table.get(i, hashes[i])
                  for i in range(self.table.depth)]
        error_estimates = self.updater.count_mean_error_estimates(self.table,
                                                                  values)
        return min(
            median(
                sorted([value - error
                        for value, error in izip(values, error_estimates)])),
            min(values))

    def most_common(self, k=None):
        """Re-query the top-n keys using the count-mean query algorithm,
        and then return the top k of the re-queried results
        :param k: The number of top items to return, at most n, as this
        function was initialized
        :return: The top k items observed by this CM sketch, and how often they
        were seen
        """
        mean_values = [(key, self.get(key)) for _, key in self.top_n]
        mean_values.sort(key=lambda pair: pair[1], reverse=True)

        if k is None or k > self.n:
            k = self.n

        return mean_values[:k]
