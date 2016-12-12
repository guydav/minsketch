# -*- coding: utf-8 -*-
from count_min_sketch import *


class LeastSquaresTopNSketch(TopNCountMinSketch):
    """
    A least-squares estimator for count-min sketches, as introduced by Lee, Lui, Yoon, & Zhang:
     https://www.usenix.org/legacy/event/imc05/tech/full_papers/lee/lee.pdf
    """
    def __init__(self, epsilon, delta, n=DEFAULT_N, table_class=ListBackedSketchTable):
        super(LeastSquaresTopNSketch, self).__init__(epsilon, delta, n, table_class)

    def most_common(self, k=None):
        """
        Generate a better estimate for the top k most-common entries using the least squares estimator
        We always solve the solution for the entire top n, and return the first k, to allow for more
        accuracy if only a subset is wanted (as the rest end up being used as noise variables
        :param self:
        :param k:
        :return:
        """
        b_vector = self.table.to_vector()
        a_matrix = numpy.zeros((self.table.depth * self.table.width, self.n + 1))

        largest = [item for _, item in self.heap.n_largest(self.n)]
        largest.reverse()

        for l in range(len(self.top_n)):
            item = largest[l]
            for table_index, hash_index in izip(range(self.table.depth), self.hash(item)):
                a_matrix[table_index * self.table.width + hash_index, l] = 1

        # Set the last column to be entirely ones
        a_matrix[:, self.n] = 1

        # Solve the equation, removing the error term
        x_vector = numpy.linalg.lstsq(a_matrix, b_vector)[0][:-1]

        new_values = zip(largest, [int(x) for x in x_vector])
        new_values.sort(key=lambda pair: pair[1], reverse=True)
        return new_values[:k]

