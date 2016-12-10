import abc
import numpy
import bitarray
import array
import struct


class SketchTable(object):
    __metaclass__ = abc.ABCMeta
    """
    An interface representing what a table for a sketch should be able to do.
    """
    @abc.abstractmethod
    def __init__(self, depth, width):
        """
        A minimal init method, saving the depth and width values
        :param depth: The depth of the list - number of tables to keep
        :param width: The width of the list - how long each list is
        """
        self.depth = depth
        self.width = width

    @abc.abstractmethod
    def get(self, depth, index):
        """
        Return the value at a certain depth (array number) and index (width)
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :return: The value at this index
        """
        raise NotImplemented("Implement this")

    @abc.abstractmethod
    def set(self, depth, index, value):
        """
        Set the value at a certain depth (array number) and index (width)
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :param value: The value to set
        :return: None
        """
        raise NotImplemented("Implement this")

    @abc.abstractmethod
    def increment(self, depth, index, value=1):
        """
        Increment the value at a certain depth (array number) and index (width)
        :param depth: The depth (which array/table) to return from
        :param index: The index within this current array
        :param value: The value to increment by, defaults to one
        :return: The new value at that index (since the caller doesn't know what it will be)
        """
        raise NotImplemented("Implement this")


class ListBackedSketchTable(SketchTable):
    """
    A naive implementation for a SketchTable, backed by a list of lists
    """
    def __init__(self, depth, width):
        super(ListBackedSketchTable, self).__init__(depth, width)
        self.table = [[0] * width for _ in range(depth)]

    def get(self, depth, index):
        return self.table[depth][index]

    def set(self, depth, index, value):
        self.table[depth][index] = value

    def increment(self, depth, index, value=1):
        self.table[depth][index] += value
        return self.table[depth][index]


class ArrayBackedSketchTable(SketchTable):
    """
    A naive implementation for a SketchTable, backed by a list of lists

    By default uses 16 bits for each counter - could be more or less based on the array.array type code used
    """
    def __init__(self, depth, width, type_code='H'):
        super(ArrayBackedSketchTable, self).__init__(depth, width)
        self.table = [array.array(type_code, (0, ) * width) for _ in range(depth)]

    def get(self, depth, index):
        return self.table[depth][index]

    def set(self, depth, index, value):
        self.table[depth][index] = value

    def increment(self, depth, index, value=1):
        self.table[depth][index] += value
        return self.table[depth][index]


class NumpyMatrixBackedSketchTable(SketchTable):
    """
    Another fairly naive implementation for a SketchTable, backed by a numpy matrix

    By default uses uint16's for each counter - could be more or less based on user desires
    """
    def __init__(self, depth, width, dtype=numpy.uint16):
        super(NumpyMatrixBackedSketchTable, self).__init__(depth, width)
        self.table = numpy.zeros((depth, width), dtype=dtype)

    def get(self, depth, index):
        return self.table[depth, index]

    def set(self, depth, index, value):
        self.table[depth, index] = value

    def increment(self, depth, index, value=1):
        self.table[depth, index] += value
        return self.table[depth, index]


SIZE_TO_STRUCT_FORMAT = {8: '<B', 16: '<H', 32: '<L', 64: '<Q'}


class BitarrayBackedSketchTable(SketchTable):
    """
    A SketchTable backed by a list of bitarrays

    By default uses 16 bits for each counter - could be more or less based on the counter_size_bits parameter
    """
    def __init__(self, depth, width, counter_size_bits=16):
        super(BitarrayBackedSketchTable, self).__init__(depth, width)
        if counter_size_bits not in SIZE_TO_STRUCT_FORMAT:
            raise ValueError('Counter size must currently be a power of two between 8 and 64')

        self.counter_size = counter_size_bits
        self.table = [bitarray.bitarray([False]) * width * counter_size_bits for _ in range(depth)]

    def get(self, depth, index):
        counter_index = self.counter_size * index
        return int(self._get_value_from_bitarray(counter_index, depth).to01(), 2)

    def _get_value_from_bitarray(self, counter_index, depth):
        return self.table[depth][counter_index:counter_index + self.counter_size]

    def set(self, depth, index, value):
        if value >= 2 ** self.counter_size:
            raise ValueError('Counter would overflow after this operation')

        counter_index = self.counter_size * index

        binary_value = bin(value)[2:].zfill(self.counter_size)
        self._set_value_to_bitarray(binary_value, counter_index, depth)

    def _set_value_to_bitarray(self, binary_value, counter_index, depth):
        self.table[depth][counter_index:counter_index + self.counter_size] = bitarray.bitarray(binary_value)

    def increment(self, depth, index, value=1):
        value += self.get(depth, index)
        self.set(depth, index, value)
        return value


class BitarrayBackedSketchTableWithStruct(SketchTable):
    """
    A SketchTable backed by a list of bitarrays

    By default uses 16 bits for each counter - could be more or less based on the counter_size_bits parameter
    """
    def __init__(self, depth, width, counter_size_bits=16):
        super(BitarrayBackedSketchTableWithStruct, self).__init__(depth, width)
        if counter_size_bits not in SIZE_TO_STRUCT_FORMAT:
            raise ValueError('Counter size must currently be a power of two between 8 and 64')

        self.counter_size = counter_size_bits
        self.struct_format = SIZE_TO_STRUCT_FORMAT[counter_size_bits]
        self.table = [bitarray.bitarray([False]) * width * counter_size_bits for _ in range(depth)]

    def get(self, depth, index):
        counter_index = self.counter_size * index
        return struct.unpack(self.struct_format, self._get_value_from_bitarray(counter_index, depth).tobytes())[0]

    def _get_value_from_bitarray(self, counter_index, depth):
        return self.table[depth][counter_index:counter_index + self.counter_size]

    def set(self, depth, index, value):
        if value >= 2 ** self.counter_size:
            raise ValueError('Counter would overflow after this operation')

        binary_value = bitarray.bitarray()
        binary_value.frombytes(struct.pack(self.struct_format, value))
        self._set_value_to_bitarray(binary_value, depth, index)

    def _set_value_to_bitarray(self, binary_value, depth, index):
        counter_index = self.counter_size * index
        self.table[depth][counter_index:counter_index + self.counter_size] = binary_value

    def increment(self, depth, index, value=1):
        value += self.get(depth, index)
        self.set(depth, index, value)
        return value


