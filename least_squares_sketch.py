# -*- coding: utf-8 -*-
"""A least-squares estimator for count-min sketches, as introduced by Lee, Lui, Yoon, & Zhang:
https://www.usenix.org/legacy/event/imc05/tech/full_papers/lee/lee.pdf
"""
from count_min_sketch import *


class LeastSquaresTopNSketch(TopNCountMinSketch):
    def most_common(self, k=None):
        """Generate a better estimate for the top k most-common entries using the least squares estimator
        We always solve the solution for the entire top n, and return the first k, to allow for more
        accuracy if only a subset is wanted (as the rest end up being used as noise variables).

        Formally, we construct $\vec{b}$ as a concatenation of the tables rows, implemented by
        table.to_vector(). $\vec{b}$ is thus (table.depth * table.width) x 1.
        We construct \textbf{A} as having the same number or rows as $\vec{b}$, and having $k + 1$ columns,
        using the following values, for $i \in \{0, 1, ..., \text{table.width} - 1\}$
        and $j \in \{0, 1, ..., \text{table.depth} - 1 \}$, we set $A_{\text{table.depth} * i + j + 1, l} = 1$
         if either the item $x_l$ is hashed into $T[i][j]$, or if $l = \text{table.width} + 1$, and otherwise
         we leave it as 0.

         We then solve using least-squares estimation (using the pseudoinverse) $A\vec{x}=\vec{b}$, sort the results
         in a descending fashion (discading the noise), and report the top k entries.
        :param k: The number of top results to return - <= to n
        :return: The least-squares estimate for how often these top k appeared
        """
        if k is None or k > self.n:
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

