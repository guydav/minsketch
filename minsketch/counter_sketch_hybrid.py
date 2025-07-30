# -*- coding: utf-8 -*-
""" This module implements a best-of-both-worlds approach - as python Counter
objects are much faster than all count-min sketches implemented, but take
indefinite amounts of space, we create a hybrid - saving new data into a
Counter as it arrives, and batch-updating the count-min sketch.
"""
from collections import Counter

DEFAULT_BATCH_SIZE = 10 ** 4


class SketchCounterHybrid(object):
    """The hybrid object, as described in the module documentation.

    On initial benchmarking, using the default batch size, it provides a
    performance boost of about 10x over using the sketch alone (on ~1M words,
    from 88 seconds to 8), at the cost about 2.5x memory (208kb to 500kb).
    """

    def __init__(self, sketch, batch_size=DEFAULT_BATCH_SIZE):
        """The sketch is initialized externally, but we expect a sketch that
        supports the most_common operation
        :param sketch: Expected to conform to the minsketch.TopNCountMinSketch
            interface
        :param batch_size: The batches in which to update the sketch
        """
        self.sketch = sketch
        self.batch_size = batch_size
        self.counter = Counter()
        self.current_batch = 0

    def _digest_counter(self):
        self.sketch.update(*zip(*self.counter.items()))
        self.counter = Counter()
        self.current_batch = 0

    def insert(self, item, count=1):
        self.counter[item] += count
        self.current_batch += count

        if self.current_batch > self.batch_size:
            self._digest_counter()

    def update(self, items, counts=None):
        if counts is None:
            [self.insert(item) for item in items]

        else:
            [self.insert(item, count) for item, count in zip(items, counts)]

    def get(self, item):
        counter_out = item in self.counter and self.counter.get(item) or 0
        return counter_out + self.sketch.get(item)

    def most_common(self, k=None):
        self._digest_counter()
        return self.sketch.most_common(k)
