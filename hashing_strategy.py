import random


ARBITRARY_LARGE_PRIME_NUMBER = 4294967291  # Largest 32 bit prime


class UniversalHashFunctionGenerator(object):
    """
    Implementation of a universal hash function family generator, as described in
    Cormen et al.'s Introduction to Algorithms.
    """
    def __init__(self, m):
        self.a_set = {None, }
        self.b_set = {None, }
        self.m = m

    def __call__(self, *args, **kwargs):
        a = None
        b = None

        while a in self.a_set and b in self.b_set:
            a = random.randint(1, ARBITRARY_LARGE_PRIME_NUMBER - 1)
            b = random.randint(0, ARBITRARY_LARGE_PRIME_NUMBER - 1)

        self.a_set.add(a)
        self.b_set.add(b)

        return lambda x: ((a * x + b) % ARBITRARY_LARGE_PRIME_NUMBER) % self.m


class NaiveHashingStrategy(object):
    def __init__(self, depth, width, hash_gen=None):
        self.depth = depth
        self.width = width

        if hash_gen is None:
            hash_gen = UniversalHashFunctionGenerator(width)

        self.hashes = [hash_gen() for _ in range(depth)]

    def __call__(self, item):
        item = hash(item)  # This handles strings well while returning itself for ints
        return [self.hashes[i](item) for i in range(self.depth)]


class DoubleHashingStrategy(object):
    """
    A single hash-pair count-min hashing scheme, based on Kirsch & Mitzenmacher (2008):
    https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf
    """
    def __init__(self, depth, width, hash_gen=None):
        self.depth = depth
        self.width = width

        if hash_gen is None:
            hash_gen = UniversalHashFunctionGenerator(ARBITRARY_LARGE_PRIME_NUMBER)

        self.hash_gen = hash_gen
        self.first_hash = hash_gen()
        self.second_hash = hash_gen()

    def __call__(self, item):
        item = hash(item)  # This handles strings well while returning itself for ints
        first = self.first_hash(item)
        second = self.second_hash(item)
        return [(first + j * second) % self.width for j in range(self.depth)]


