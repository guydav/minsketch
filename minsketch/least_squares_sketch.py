# -*- coding: utf-8 -*-
"""A least-squares estimator for count-min sketches, as introduced by Lee, Lui,
Yoon, & Zhang:
https://www.usenix.org/legacy/event/imc05/tech/full_papers/lee/lee.pdf
"""
from itertools import izip

import numpy

import count_min_sketch


class LeastSquaresTopNSketch(count_min_sketch.TopNCountMinSketch):
    """The least-squares top-N sketch only overrides the most common item
    finding heuristic:
    """

    def most_common(self, k=None):
        r"""Generate a better estimate for the top k most-common entries using
        the least squares estimator. We always solve the solution for the
        entire top n, and return the first k, to allow for more accuracy if
        only a subset is wanted (as the rest end up being used as noise
        variables).

        Formally, we construct :math:`\vec{b}` as a concatenation of the tables
        rows, implemented by table.to_vector(). The size of :math:`\vec{b}` is
        thus (table.depth * table.width) x 1.

        We construct :math:`A` as having the same number or rows as
        :math:`\vec{b}`, and having :math:`k + 1` columns, using the following
        values:

        for :math:`i \in \{0, 1, ..., \text{table.width} - 1\}`
        and :math:`j \in \{0, 1, ..., \text{table.depth} - 1 \}`,

        we set \ :math:`A_{\text{table.depth} * i + j + 1, l} = 1` if either
        the item :math:`x_l` is hashed into :math:`T[i][j]`, or if
        :math:`l = \text{table.width} + 1`, and otherwise we leave it as 0.

        We then solve using least-squares estimation (using the pseudoinverse)
        :math:`A\vec{x}=\vec{b}`, sort the results in a descending fashion
        (discarding the noise), and report the top k entries.

        :param k: The number of top results to return - <= to n
        :return: The least-squares estimate for how often these top k appeared
        """
        if k is None or k > self.n:
            k = self.n

        b_vector = self.table.to_vector()
        a_matrix = numpy.zeros(
            (self.table.depth * self.table.width, self.n + 1))

        largest = [item for _, item in self.heap.n_largest(self.n)]
        largest.reverse()

        for l in range(len(self.top_n)):
            item = largest[l]
            for table_index, hash_index in izip(
                    range(self.table.depth), self.hash(item)):
                a_matrix[table_index * self.table.width + hash_index, l] = 1

        # Set the last column to be entirely ones
        a_matrix[:, self.n] = 1

        # Solve the equation, removing the error term
        x_vector = numpy.linalg.lstsq(a_matrix, b_vector)[0][:-1]

        new_values = zip(largest, [int(x) for x in x_vector])
        new_values.sort(key=lambda pair: pair[1], reverse=True)
        return new_values[:k]
