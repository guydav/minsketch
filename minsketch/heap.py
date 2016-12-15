# -*- coding: utf-8 -*-
"""An OOP wrapper of Python's built-in heapq - a min heap
"""

import heapq


class MinHeap(object):
    def __init__(self, data=None):
        if data is not None:
            self.heap = heapq.heapify(data)

        else:
            self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def pop(self):
        return heapq.heappop(self.heap)

    def push_pop(self, item):
        return heapq.heappushpop(self.heap, item)

    def n_smallest(self, n):
        return heapq.nsmallest(n, self.heap)

    def n_largest(self, n):
        return heapq.nsmallest(n, self.heap)

    def peek(self):
        return self.heap[0]

    def heapify(self):
        heapq.heapify(self.heap)

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str(self.heap)

    def __getitem__(self, item):
        return self.heap.__getitem__(item)
