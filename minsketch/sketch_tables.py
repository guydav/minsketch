# -*- coding: utf-8 -*-
"""Implementation of the different table styles. Four are currently implemented
- backing by a simple python list, backing by python arrays (which are more
efficient, but provide a very similar interface), backing by numpy matrices
(which turned out to be rather inefficient), and backing by bitarrays (which
saves a lot of memory, but incurs a significant performance penalty.

In the future I should implement a table in C, bridge it to Python, and call it
a day.
"""
import abc
import numpy
import bitarray
import array
import struct
from functools import reduce

POSITIVE_INFINITY = float('inf')


class SketchTable(object):
    __metaclass__ = abc.ABCMeta
    """An interface representing what a table for a sketch should be able to
    do. It actually also handles the entire implementation for things that
    conform to the python list interface.
    """

    @abc.abstractmethod
    def __init__(self, depth, width):
        """A minimal init method, saving the depth and width values
        :param depth: The depth of the list - number of tables to keep
        :param width: The width of the list - how long each list is
        """
        self.depth = depth
        self.width = width
        self.table = None
        self.total = 0

    def get(self, depth, index):
        """Return the value at a certain depth (array number) and index (width)
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :return: The value at this index
        """
        return self.table[depth][index]

    def set(self, depth, index, value):
        """Set the value at a certain depth (array number) and index (width)
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :param value: The value to set
        :return: None
        """
        self.table[depth][index] = value

    def increment(self, depth, index, value=1):
        """Increment the value at a certain depth (array number) and index
        (width). Also tracks how many total values have been inserted.
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :param value: The value to increment by, defaults to one
        :return: The new value at that index (since the caller doesn't know
        what it will be)
        """
        self.total += value
        self.table[depth][index] += value
        return self.table[depth][index]

    def to_vector(self):
        """Transform the internal table to a single-dimensional vector
        :return: A vector containing the entire table
        """
        return numpy.array(reduce(lambda a, b: a + b, self.table))

    def __iter__(self):
        """A naive iteration strategy, assuming that self.table is iterable,
        and that its rows are also iterable. Subclasses should override
        if a different iteration strategy is preferable, or this one
        is insensible
        :return: Yielding one item at a time.
        """
        for row in self.table:
            for item in row:
                yield item

    def row_iter(self):
        """A by-row iterator, written to help a specific use-case:
        count-mean-min with conservative update (CMM-CU)'s summation over each
        row
        :return: Yielding one row at a time
        """
        for row in self.table:
            yield row

    def decrement_all(self, upper_bound=None, lower_bound=0):
        """A decrement-all operation designed to support lossy counting - see
        Goyal and Daumé (2011):
        http://www.umiacs.umd.edu/~amit/Papers/goyalLCUSketchAAAI11.pdf

        When this function is called, it iterates through all items, and
        decrements whichever match the bounds. The un-intuitive ordering of
        arguments stems from the fact that in many cases it makes sense to
        change the upper bound (LCU-1, LCU-WS, LCU-SWS), while none I've seen
        so far attempt to change the lower bound.
        :param upper_bound: The upper bound, inclusive, to decrement in
        :param lower_bound: The lower bound, exclusive, to decrement in
        :return: None; all relevant items decremented
        """
        if upper_bound is None:
            upper_bound = POSITIVE_INFINITY

        for item in self:
            if lower_bound < item <= upper_bound:
                item -= 1


class ListBackedSketchTable(SketchTable):
    """A naive implementation for a SketchTable, backed by a list of lists
    """

    def __init__(self, depth, width):
        super(ListBackedSketchTable, self).__init__(depth, width)
        self.table = [[0] * width for _ in range(depth)]


class ArrayBackedSketchTable(SketchTable):
    """A naive implementation for a SketchTable, backed by a list of lists

    By default uses 16 bits for each counter - could be more or less based on
    the array.array type code used
    """

    def __init__(self, depth, width, type_code='H'):
        super(ArrayBackedSketchTable, self).__init__(depth, width)
        self.table = [array.array(type_code, (0, ) * width)
                      for _ in range(depth)]


class NumpyMatrixBackedSketchTable(SketchTable):
    """Another fairly naive implementation for a SketchTable, backed by a numpy
    matrix

    By default uses uint16's for each counter - could be more or less based on
    user desires
    """

    def __init__(self, depth, width, dtype=numpy.uint16):
        super(NumpyMatrixBackedSketchTable, self).__init__(depth, width)
        self.table = numpy.zeros((depth, width), dtype=dtype)

    def to_vector(self):
        return numpy.reshape(self.table, -1)

    def __iter__(self):
        return numpy.nditer(self.table)


SIZE_TO_STRUCT_FORMAT = {8: 'B', 16: 'H', 32: 'L', 64: 'Q'}


class BitarrayBackedSketchTable(SketchTable):
    """A SketchTable backed by a list of bitarrays

    By default uses 16 bits for each counter - could be more or less based on
    the counter_size_bits parameter
    """

    def __init__(self, depth, width, counter_size_bits=16):
        super(BitarrayBackedSketchTable, self).__init__(depth, width)
        if counter_size_bits not in SIZE_TO_STRUCT_FORMAT:
            raise ValueError(
                'Counter size must be a power of two between 8 and 64')

        self.counter_size = counter_size_bits
        self.struct_format = SIZE_TO_STRUCT_FORMAT[counter_size_bits]
        self.table = [bitarray.bitarray([False]) * width * counter_size_bits
                      for _ in range(depth)]

    def get(self, depth, index):
        return self._bits_to_number(self._get_value_from_bits(index, depth))

    def set(self, depth, index, value):
        if value >= 2**self.counter_size:
            raise ValueError('Counter would overflow after this operation')

        binary_value = self._to_bits(value)
        self._set_value_to_bits(binary_value, depth, index)

    def increment(self, depth, index, value=1):
        self.total += 1
        value += self.get(depth, index)
        self.set(depth, index, value)
        return value

    def to_vector(self):
        unpacked_rows = [
            struct.unpack(self.struct_format * self.width, table_row)
            for table_row in self.table
        ]
        return numpy.array(reduce(lambda a, b: a + b, unpacked_rows))

    def decrement_all(self, upper_bound=POSITIVE_INFINITY, lower_bound=0):
        """Overriding the default implementation to avoid packing and unpacking
        whenever unnecessary
        :param upper_bound: The upper bound, inclusive, to decrement in
        :param lower_bound: The lower bound, exclusive, to decrement in
        :return: None; all relevant items decremented
        """
        # TODO: If feeling adventurous, benchmark against unpacking all values,
        # and then decrementing the necessary ones
        lower_bound_bits = self._to_bits(lower_bound)

        if upper_bound is None or upper_bound == POSITIVE_INFINITY:
            upper_bound = (2**self.counter_size) - 1

        upper_bound_bits = self._to_bits(upper_bound)

        for depth in range(self.depth):
            for index in range(self.width):
                value = self._get_value_from_bits(index, depth)
                if lower_bound_bits < value <= upper_bound_bits:
                    self.set(depth, index, value - 1)

    def __iter__(self):
        for row in self.row_iter():
            for item in row:
                yield item

    def row_iter(self):
        for row in self.table:
            yield struct.unpack(self.struct_format * self.width, row)

    def _bits_to_number(self, bits):
        return struct.unpack(self.struct_format, bits.tobytes())[0]

    def _get_value_from_bits(self, index, depth):
        counter_index = self.counter_size * index
        return self.table[depth][counter_index:counter_index +
                                 self.counter_size]

    def _to_bits(self, value):
        binary_value = bitarray.bitarray()
        binary_value.frombytes(struct.pack(self.struct_format, value))
        return binary_value

    def _set_value_to_bits(self, binary_value, depth, index):
        counter_index = self.counter_size * index
        self.table[depth][counter_index:counter_index +
                          self.counter_size] = binary_value
