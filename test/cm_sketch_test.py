# -*- coding: utf-8 -*-
from double_hashing import *
from least_squares_sketch import LeastSquaresTopNSketch

import re
from collections import Counter
from functools import partial
from timeit import timeit
from asizeof import asizeof
from sortedcontainers import SortedDict
from tabulate import tabulate
import copy


def test():
    count = 100000
    numbers = [random.randint(1, ARBITRARY_LARGE_PRIME_NUMBER - 1) for _ in range(count)]
    counts = [random.randint(10, 1000) for _ in range(count)]

    cms = TopNCountMinSketch(10 ** -7, 0.005)

    for (number, count) in izip(numbers, counts):
        cms.insert(number, count)

    print(cms.table.depth, cms.table.width, cms.table.depth * cms.table.width)

    total_error = 0
    percent_error = 0

    for i in range(count):
        error = float(cms.get(numbers[i]) - counts[i])
        total_error += error
        percent_error = error / counts[i]

    print('Total error:', total_error / sum(counts))
    print('Average percent error:', percent_error / count)
    print(cms)


def text_test(text_file_name, epsilon, delta):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    sketches = SortedDict(
        list=TopNCountMinSketch(delta, epsilon, table_class=ListBackedSketchTable),
        array=TopNCountMinSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        matrix=TopNCountMinSketch(delta, epsilon, table_class=NumpyMatrixBackedSketchTable),
        bitarray=TopNCountMinSketch(delta, epsilon, table_class=BitarrayBackedSketchTable),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches)


def benchmark(words, sketches):
    results = SortedDict()

    results['Insertion time'] = run_single_benchmark(
        sketches, lambda sketch: timeit(partial(sketch.update, words), number=1))

    results['Memory footprint'] = run_single_benchmark(sketches, asizeof)

    # TODO: Deal with aligning everything to the right
    header = ['Metric'] + list(sketches.keys())
    print(tabulate([[metric] + results[metric] for metric in results], header,
                   numalign='decimal', tablefmt='fancy_grid'))


def run_single_benchmark(sketches, benchmark_func):
    return [benchmark_func(sketch) for sketch in sketches.values()]


def most_common_comparison(sketches, n):
    # top_10s = run_single_benchmark(sketches,
    #                                lambda sketch: sketch.most_common(n / 2))

    counter = sketches['counter']
    mc = {key : value for key, value in counter.most_common(n)}
    top_n_by_sketch = SortedDict()
    for name in sketches.keys():
        ordered_results = SortedDict()
        for key in mc:
            ordered_results[key] = sketches[name].get(key)

        top_n_by_sketch[name] = ordered_results

    for top_n in top_n_by_sketch.values():
        for key in top_n:
            top_n[key] -= counter[key]

    diff_results = [[key] + [top_n[key] for top_n in top_n_by_sketch.values()]
                    for key in mc]
    header = ['Item'] + list(sketches.keys())
    print(tabulate(diff_results, header, tablefmt='fancy_grid'))

    top_n_percent_over = copy.deepcopy(top_n_by_sketch)
    for top_n in top_n_percent_over.values():
        for key in top_n:
            top_n[key] /= float(counter[key])

    percent_results = [[key] + [top_n[key] for top_n in top_n_percent_over.values()]
                       for key in mc]
    percent_results.append(['Sum Total'] + [sum(top_n.values()) for top_n in top_n_percent_over.values()])
    print(tabulate(percent_results, header, tablefmt='fancy_grid'))


def test_double_hashing(text_file_name, epsilon, delta, n=DEFAULT_N):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    sketches = SortedDict(
        array=TopNCountMinSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        array_hash_pair=HashPairCMSketch(delta=n, epsilon=epsilon, table_class=ArrayBackedSketchTable),
        array_hash_pair_multi=MultiHashPairTopNCMSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_to_vector():
    depth = 2
    width = 10
    count = depth * width
    counts = [random.randint(1, 10) for _ in xrange(count)]

    vectors = []
    for table_class in (ListBackedSketchTable, ArrayBackedSketchTable,
                        NumpyMatrixBackedSketchTable, BitarrayBackedSketchTable):
        table = table_class(depth, width)
        for d in range(depth):
            for i in range(width):
                table.set(d, i, counts[d * width + i])

        vectors.append(table.to_vector())

    for v in vectors:
        print(v)


def test_least_squares(text_file_name, epsilon, delta, n=DEFAULT_N):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    sketches = SortedDict(
        array=TopNCountMinSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        array_hash_pair=HashPairCMSketch(delta=n, epsilon=epsilon, table_class=ArrayBackedSketchTable),
        array_hash_pair_multi=MultiHashPairTopNCMSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        least_squares=LeastSquaresTopNSketch(delta, epsilon, table_class=ArrayBackedSketchTable),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_update_strategies(text_file_name, epsilon, delta, n=DEFAULT_N):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    # Using 5 * n here to account for different values in the exact top n
    n *= 5

    sketches = SortedDict(
        array=TopNCountMinSketch(delta, epsilon, n=n, table_class=ArrayBackedSketchTable),
        array_hash_pair=HashPairCMSketch(delta, epsilon, n=n, table_class=ArrayBackedSketchTable),
        array_conservative=TopNCountMinSketch(delta, epsilon, n=n,
                                              table_class=ArrayBackedSketchTable,
                                              update_strategy=ConservativeUpdateStrategy),
        array_hash_pair_conservative=HashPairCMSketch(delta, epsilon, n=n,
                                                      table_class=ArrayBackedSketchTable,
                                                      update_strategy=ConservativeUpdateStrategy),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)

if __name__ == '__main__':
    # text_test('shakespeare.txt', 0.001, 10 ** -7)
    # test_double_hashing('shakespeare.txt', 0.001, 10 ** -5, 20)
    # test_to_vector()
    # test_least_squares('shakespeare.txt', 0.001, 10 ** -5, 10)
    test_update_strategies('shakespeare.txt', 0.01, 10 ** -4, 10)
