# -*- coding: utf-8 -*-
from itertools import izip


class NaiveUpdateStrategy(object):
    """
    This could clearly be a function, rather than a class. Why make it a class, then?
    Two reasons:
    1) Compatibility with other strategies, all of whom are classes
    2) In the event I'll want to implement stateful updating strategies in the future
    """
    def __call__(self, table, hashes, count):
        """
        The most naive possible updating strategy - always update by everything
        :param table: The table being updated
        :param hashes: The hashes of the current item, each per row in the table
        :param count: The count to increment by
        :return: The new minimal count for this item being updated
        """
        return min([table.increment(i, hashes[i], count) for i in range(table.depth)])


class ConservativeUpdateStrategy(object):
    """
    A conservative update strategy, as introduced by Cormode (2009) and Goyal (2010):
    http://www.cs.utah.edu/~amitg/papers/goyal10GEMS.pdf
    At every point, only update the minimal number of items.

    As for why this is a class, see the class-comment for NaiveUpdateStrategy
    """
    def __call__(self, table, hashes, count):
        """
        This only supports positive counts - with negative counts, this approach is
        unsensible, and might lead to unexpected results.
        :param table: The table being updated
        :param hashes: The hashes of the current item, each per row in the table
        :param count: The count to increment by
        :return: The new minimal count for this item being updated
        """
        if 0 > count:
            raise ValueError('Conservative updating strategy does not support negative counts')

        current_values = [table.get(i, hashes[i]) for i in range(table.depth)]
        new_current_min = min(current_values) + count
        [table.set(i, hashes[i], new_current_min)
         for i, value in izip(range(table.depth), current_values)
         if value < new_current_min]

        return new_current_min
