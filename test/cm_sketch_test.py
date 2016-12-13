# -*- coding: utf-8 -*-
"""A testing / benchmarking code. This does not provide any regression or unit tests, at least not so far.
It does provide basic system tests, and some measure of benchmarks, of the different configurations possible.

This is not the prettiest code ever written; but it works, and provides an easy way to play around with the code.
"""
from double_hashing import *
from least_squares_sketch import LeastSquaresTopNSketch
from count_mean_sketch import CountMeanMinSketch, HashPairCountMeanMinSketch

import re
import copy
from collections import Counter, OrderedDict
from functools import partial
from timeit import timeit
from asizeof import asizeof
from tabulate import tabulate
from numpy import mean


COUNTER_KEY = 'counter'


def basic_count_min_test():
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


def read_words(text_file_name):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    return words


def text_test(text_file_name, epsilon, delta, n=DEFAULT_N, n_factor=5):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    n *= n_factor
    sketches = OrderedDict(
        list=TopNCountMinSketch(delta, epsilon, n=n, table_class=ListBackedSketchTable),
        array=TopNCountMinSketch(delta, epsilon, n=n, table_class=ArrayBackedSketchTable),
        matrix=TopNCountMinSketch(delta, epsilon, n=n, table_class=NumpyMatrixBackedSketchTable),
        bitarray=TopNCountMinSketch(delta, epsilon, n=n, table_class=BitarrayBackedSketchTable),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n / n_factor)


def benchmark(words, sketches):
    results = OrderedDict()

    results['Insertion time'] = run_single_benchmark(
        sketches, lambda sketch: timeit(partial(sketch.update, words), number=1))

    results['Memory footprint'] = run_single_benchmark(sketches, asizeof)

    results['Depth'] = []
    results['Width'] = []
    for key in sketches:
        sk = sketches[key]
        if isinstance(sk, Counter):
            results['Depth'].append('N/A')
            results['Width'].append('N/A')

        elif isinstance(sk, MultiHashPairTopNCMSketch):
            results['Depth'].append(sk.sketches[0].table.depth)
            results['Width'].append(sk.sketches[0].table.width)

        else:
            results['Depth'].append(sk.table.depth)
            results['Width'].append(sk.table.width)

    header = ['Metric'] + list(sketches.keys())
    print(tabulate([[metric] + results[metric] for metric in results], header,
                   numalign='decimal', tablefmt='fancy_grid'))


def run_single_benchmark(sketches, benchmark_func):
    return [benchmark_func(sketch) for sketch in sketches.values()]


def most_common_comparison(sketches, n):
    # top_10s = run_single_benchmark(sketches,
    #                                lambda sketch: sketch.most_common(n / 2))

    counter = sketches[COUNTER_KEY]
    mc = {key : value for key, value in counter.most_common(n)}
    top_n_by_sketch = OrderedDict()
    for name in sketches.keys():
        ordered_results = OrderedDict()
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


def test_double_hashing(text_file_name, epsilon, delta, n=DEFAULT_N, table_class=ArrayBackedSketchTable):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    sketches = OrderedDict(
        array=TopNCountMinSketch(delta, epsilon, table_class=table_class),
        array_hash_pair=HashPairCMSketch(delta=n, epsilon=epsilon, table_class=table_class),
        array_hash_pair_multi=MultiHashPairTopNCMSketch(delta, epsilon, table_class=table_class),
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


def test_least_squares(text_file_name, epsilon, delta, n=DEFAULT_N, table_class=ArrayBackedSketchTable):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))

    sketches = OrderedDict(
        array=TopNCountMinSketch(delta, epsilon, table_class=table_class),
        array_hash_pair=HashPairCMSketch(delta=n, epsilon=epsilon, table_class=table_class),
        array_hash_pair_multi=MultiHashPairTopNCMSketch(delta, epsilon, table_class=table_class),
        least_squares=LeastSquaresTopNSketch(delta, epsilon, table_class=table_class),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_update_strategies(text_file_name, epsilon, delta, n=DEFAULT_N, table_class=ArrayBackedSketchTable):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    # Using 5 * n here to account for different values in the exact top n
    n *= 5

    sketches = OrderedDict(
        array=TopNCountMinSketch(delta, epsilon, n=n, table_class=table_class),
        array_hash_pair=HashPairCMSketch(delta, epsilon, n=n, table_class=table_class),
        array_conservative=TopNCountMinSketch(delta, epsilon, n=n,
                                              table_class=table_class,
                                              update_strategy=ConservativeUpdateStrategy),
        array_hash_pair_conservative=HashPairCMSketch(delta, epsilon, n=n,
                                                      table_class=table_class,
                                                      update_strategy=ConservativeUpdateStrategy),
        counter=Counter())

    # This test does generate the hash-pair ones and the non-hash-pair ones with different sizes
    # However, forcing the non-hash-pair ones into the hash-pair scheme only generates worse results

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_count_mean(text_file_name, epsilon, delta, n=DEFAULT_N, table_class=ArrayBackedSketchTable):
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    # Using 5 * n here to account for different values in the exact top n
    n *= 5

    sketches = OrderedDict(
        hash_pair=HashPairCMSketch(delta, epsilon, n=n, table_class=table_class),
        hash_pair_cons=HashPairCMSketch(delta, epsilon, n=n,
                                        table_class=table_class,
                                        update_strategy=ConservativeUpdateStrategy),
        count_mean=CountMeanMinSketch(delta, epsilon, n=n, table_class=table_class),
        hash_pair_count_mean=HashPairCountMeanMinSketch(delta, epsilon, n=n, table_class=table_class),
        count_mean_cons=CountMeanMinSketch(delta, epsilon, n=n,
                                           table_class=table_class,
                                           update_strategy=ConservativeUpdateStrategy),
        hash_pair_count_mean_cons=HashPairCountMeanMinSketch(delta, epsilon, n=n,
                                                             table_class=table_class,
                                                             update_strategy=ConservativeUpdateStrategy),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_lossy_strategy(text_file_name, epsilon, delta, gamma=0.01, bucket_size=100, n=DEFAULT_N,
                        table_class=ArrayBackedSketchTable):
    # To test lossy strategies, we process the files similarly as before
    # And compare by buckets, akin to Goyal and Daumé (2010)
    words = re.findall(r'\w+', open(text_file_name).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    # Using 5 * n here to account for different values in the exact top n
    counter = Counter()
    sketches = OrderedDict(
        no_lossy=TopNCountMinSketch(delta, epsilon, n=n, table_class=table_class),
        lossy_no_threshold=TopNCountMinSketch(delta, epsilon, n=n,
                                              table_class=table_class,
                                              lossy_strategy=LossyUpdateStrategy(gamma, no_threshold_func)),
        lossy_1_threshold=TopNCountMinSketch(delta, epsilon, n=n,
                                             table_class=table_class,
                                             lossy_strategy=LossyUpdateStrategy(gamma, one_threshold_func)),
        lossy_window_size=TopNCountMinSketch(delta, epsilon, n=n,
                                             table_class=table_class,
                                             lossy_strategy=LossyUpdateStrategy(gamma, window_size_threshold_func)),
        lossy_sqrt_window=TopNCountMinSketch(delta, epsilon, n=n,
                                             table_class=table_class,
                                             lossy_strategy=LossyUpdateStrategy(gamma,
                                                                                sqrt_window_size_threshold_func)),
        counter=counter)

    benchmark(words, sketches)
    benchmark_by_buckets(bucket_size, sketches)


def benchmark_by_buckets(bucket_size, sketches):
    counter = sketches[COUNTER_KEY]
    results_by_bucket = [{key: [] for key in sketches if key != COUNTER_KEY}
                         for _ in range(counter.most_common(1)[0][1] / bucket_size + 1)]
    for item in counter:
        for key in sketches:
            if key == COUNTER_KEY:
                continue
            count = counter[item]

            results_by_bucket[count / bucket_size][key].append((sketches[key].get(item) - count))

    # Turn into average
    results_table = []
    for i in range(len(results_by_bucket)):
        start = i * bucket_size
        bucket = '{start}-{end}'.format(start=start, end=start + bucket_size - 1)
        bucket_results = [results_by_bucket[i][key] for key in sketches if key != COUNTER_KEY]

        if any(bucket_results):
            results_table.append([bucket] + [mean(br) for br in bucket_results])

    results_table.append(['Overall average'] + [mean([results_table[row][col] for row in range(len(results_table))])
                                                for col in range(1, len(results_table[0]))])
    header = ['Bucket'] + [key for key in sketches if key != COUNTER_KEY]
    print(tabulate(results_table, header, tablefmt='fancy_grid'))


def test_lossy_conservative_strategy(text_file_name, epsilon, delta, gamma=0.01, bucket_size=100, n=DEFAULT_N,
                                     table_class=ArrayBackedSketchTable):
    # To test lossy strategies, we process the files similarly as before
    # And compare by buckets, akin to Goyal and Daumé (2010)
    words = read_words(text_file_name)
    # Using 5 * n here to account for different values in the exact top n
    counter = Counter()
    sketches = OrderedDict(
        no_lossy=TopNCountMinSketch(delta, epsilon, n=n, table_class=table_class),
        no_lossy_cons=TopNCountMinSketch(delta, epsilon, n=n,
                                         table_class=table_class,
                                         update_strategy=ConservativeUpdateStrategy),
        lossy_no_threshold=TopNCountMinSketch(delta, epsilon, n=n,
                                              table_class=table_class,
                                              lossy_strategy=LossyUpdateStrategy(gamma, no_threshold_func)),
        lossy_no_threshold_cons=TopNCountMinSketch(delta, epsilon, n=n,
                                                   table_class=table_class,
                                                   update_strategy=ConservativeUpdateStrategy,
                                                   lossy_strategy=LossyUpdateStrategy(gamma, no_threshold_func)),
        lossy_window_size=TopNCountMinSketch(delta, epsilon, n=n,
                                             table_class=table_class,
                                             lossy_strategy=LossyUpdateStrategy(gamma, window_size_threshold_func)),
        lossy_sqrt_window_cons=TopNCountMinSketch(delta, epsilon, n=n,
                                                  table_class=table_class,
                                                  update_strategy=ConservativeUpdateStrategy,
                                                  lossy_strategy=LossyUpdateStrategy(gamma,
                                                                                     window_size_threshold_func)),
        counter=counter)

    benchmark(words, sketches)
    benchmark_by_buckets(bucket_size, sketches)


if __name__ == '__main__':
    text_test('shakespeare.txt', 0.001, 10 ** -7)
    test_double_hashing('shakespeare.txt', 0.001, 10 ** -5, 20)
    test_to_vector()
    test_least_squares('shakespeare.txt', 0.001, 10 ** -5, 10)
    test_update_strategies('shakespeare.txt', 0.01, 10 ** -4, 10)
    test_count_mean('shakespeare.txt', 0.01, 10 ** -4, 10)
    test_lossy_strategy('shakespeare.txt', 0.001, 10 ** -5)
    test_lossy_conservative_strategy('shakespeare.txt', 0.001, 10 ** -5)
    # TODO: implement joint-counter/CMS
    # TODO: write documentation

