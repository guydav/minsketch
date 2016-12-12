# -*- coding: utf-8 -*-
from count_min_sketch import *


class LeastSquaresTopNSketch(TopNCountMinSketch):
    """
    A least-squares estimator for count-min sketches, as introduced by Lee, Lui, Yoon, & Zhang:
     https://www.usenix.org/legacy/event/imc05/tech/full_papers/lee/lee.pdf
    """
    def __init__(self, delta, epsilon, n=DEFAULT_N,
                 table_class=ListBackedSketchTable,
                 hash_strategy=NaiveHashingStrategy,
                 update_strategy=NaiveUpdateStrategy,
                 lossy_strategy=NoLossyUpdateStrategy):
        super(LeastSquaresTopNSketch, self).__init__(delta, epsilon, n=n,
                                                     table_class=table_class,
                                                     hash_strategy=hash_strategy,
                                                     update_strategy=update_strategy,
                                                     lossy_strategy=NoLossyUpdateStrategy)

    def most_common(self, k=None):
        """
        Generate a better estimate for the top k most-common entries using the least squares estimator
        We always solve the solution for the entire top n, and return the first k, to allow for more
        accuracy if only a subset is wanted (as the rest end up being used as noise variables
        :param k: The number of top results to return - <= to n
        :return:
        """
        if k is None or  k > self.n:
            k = self.n

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

