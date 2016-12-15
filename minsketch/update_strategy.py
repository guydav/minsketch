# -*- coding: utf-8 -*-
"""An implementation of naive and conservative update strategies.
Each update strategy also implements an error-estimate, used by count-min-mean.
"""
from itertools import izip


class NaiveUpdateStrategy(object):
    """ This could clearly be a function, rather than a class. Why make it a
    class, then? Two reasons:
    1) Compatibility with other strategies, all of whom are classes
    2) In the event I'll want to implement stateful updating strategies in the
    future
    3) I already ended up using its class-ness, to couple error estimation
    strategies here.
    """

    def __call__(self, table, hashes, count):
        """The most naive possible updating strategy - always update by
        everything
        :param table: The table being updated
        :param hashes: The hashes of the current item, each per row in the
        table
        :param count: The count to increment by
        :return: The new minimal count for this item being updated
        """
        return min(
            [table.increment(i, hashes[i], count) for i in range(table.depth)])

    def count_mean_error_estimates(self, table, values):
        """Provide an error estimate, using the total insertions into the table
        as a baseline
        :param table: The table being updated
        :param values: Values of the item we return error estimates for
        :return: Error estimates for each of these values
        """
        baselines = [table.total] * table.depth
        return self._error_estimates(table, values, baselines)

    def _error_estimates(self, table, values, baselines):
        return [(baseline - value) / (table.width - 1)
                for value, baseline in izip(values, baselines)]


class ConservativeUpdateStrategy(NaiveUpdateStrategy):
    """A conservative update strategy, as introduced by Cormode (2009) and
    Goyal (2010):
    http://www.cs.utah.edu/~amitg/papers/goyal10GEMS.pdf
    At every point, only update the minimal number of items.

    As for why this is a class, see the class-comment for NaiveUpdateStrategy
    """

    def __call__(self, table, hashes, count):
        """
        This only supports positive counts - with negative counts, this
        approach is insensible, and might lead to unexpected results.
        :param table: The table being updated
        :param hashes: The hashes of the current item, each per row in the
        table
        :param count: The count to increment by
        :return: The new minimal count for this item being updated
        """
        if 0 > count:
            raise ValueError(
                'Conservative updating does not support negative counts')

        current_values = [table.get(i, hashes[i]) for i in range(table.depth)]
        new_current_min = min(current_values) + count
        [table.set(i, hashes[i], new_current_min)
         for i, value in izip(range(table.depth), current_values)
         if value < new_current_min]

        return new_current_min

    def count_mean_error_estimates(self, table, values):
        """Error estimates for the count-min-mean, but in this case, we used
        the row-wise sum as an estimator for the entry in that row, rather than
        the entire total.
        :param table: The table being updated
        :param values: Values of the item we return error estimates for
        :return: Error estimates for each of these values
        """
        baselines = [sum(row) for row in table.row_iter()]
        return self._error_estimates(table, values, baselines)
