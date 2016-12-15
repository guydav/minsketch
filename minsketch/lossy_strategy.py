# -*- coding: utf-8 -*-
"""Implementation of lossy tracking strategies.
"""
from sketch_tables import POSITIVE_INFINITY


class NoLossyUpdateStrategy(object):
    """
    This looks silly, but it's implemented as a default over always doing a
    lossy update strategy check.
    """

    def __call__(self, table=None):
        pass


class LossyUpdateStrategy(object):
    """
    A simple class implementing lossy update strategies
    """

    def __init__(self, gamma, threshold_func):
        """
        :param gamma: The lossy parameter, where each window is 1/gamma items
        :param threshold_func: A function returning a threshold from a number
        of windows seen so far
        """
        self.gamma = gamma
        self.window_size = 1.0 / gamma
        self.count = 0
        self.window_count = 0
        self.threshold_func = threshold_func

    def __call__(self, table):
        self.count += 1

        if self.count > self.window_size:
            self.count = 0
            self.window_count += 1
            table.decrement_all(self.threshold_func(self.window_count))


def no_threshold_func(window_count=None):
    """The threshold function for LCU-ALL - accepts a parameter to conform to
    the 'interface'
    :param window_count: How many windows have appeared so far
    :return: positive infinity, always
    """
    return POSITIVE_INFINITY


def one_threshold_func(window_count=None):
    """The threshold function for LCU-1 - accepts a parameter to conform to the
    'interface'
    :param window_count: How many windows have appeared so far
    :return: 1, always
    """
    return 1


def window_size_threshold_func(window_count):
    """The threshold function for LCU-WS
    :param window_count: How many windows have appeared so far
    :return: the count of the windows seen so far as a threshold
    """
    return window_count


def sqrt_window_size_threshold_func(window_count):
    """The threshold function for LCU-SWS
    :param window_count: How many windows have appeared so far
    :return: the square root of the windows seen so far as a threshold
    """
    return window_count**0.5
