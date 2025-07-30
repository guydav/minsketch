# -*- coding: utf-8 -*-
"""Testing / benchmarking code. This does not provide any regression or unit
tests, at least not so far. It does provide basic system tests, and some
measure of benchmarks, of the different configurations possible.

This is not the prettiest code ever written; but it works, and provides an easy
way to play around with the code.
"""
import copy
import random
import re
import os.path
from collections import Counter, OrderedDict
from functools import partial

from timeit import timeit
from asizeof import asizeof
from numpy import mean
from tabulate import tabulate

from minsketch import count_mean_sketch, count_min_sketch, double_hashing, \
    hash_strategy, least_squares_sketch, lossy_strategy, sketch_tables, \
    update_strategy, counter_sketch_hybrid

COUNTER_KEY = 'counter'
DEFAULT_TEST_DATA_FILE = 'hamlet.txt'
DEFAULT_EPSILON = 0.001
DEFAULT_DELTA = 10 ** -4
DEFAULT_N = 20


def read_words(test_data_file):
    path_with_dir = os.path.join(os.path.dirname(__file__), test_data_file)
    words = re.findall(r'\w+', open(path_with_dir).read().lower())
    print('{length} words in the document'.format(length=len(words)))
    return words


def basic_count_min_test():
    count = 100000
    numbers = [
        random.randint(1, hash_strategy.ARBITRARY_LARGE_PRIME_NUMBER - 1)
        for _ in range(count)
    ]
    counts = [random.randint(10, 1000) for _ in range(count)]

    cms = count_min_sketch.TopNCountMinSketch(10e-7, 0.005)

    for (number, count) in zip(numbers, counts):
        cms.insert(number, count)

    print(cms.table.depth, cms.table.width, cms.table.depth * cms.table.width)

    total_error = 0
    percent_error = 0

    for i in range(count):
        error = float(cms.get(numbers[i]) - counts[i])
        total_error += error
        percent_error += error / counts[i]

    print('Total error:', total_error / sum(counts))
    print('Average percent error:', percent_error / count)
    print(cms)


def text_test(words_file=DEFAULT_TEST_DATA_FILE,
              epsilon=DEFAULT_EPSILON,
              delta=DEFAULT_DELTA,
              n=count_min_sketch.DEFAULT_N,
              n_factor=5):

    words = read_words(words_file)

    n *= n_factor
    sketches = OrderedDict(
        list=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=sketch_tables.ListBackedSketchTable),
        array=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=sketch_tables.ArrayBackedSketchTable),
        matrix=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=sketch_tables.NumpyMatrixBackedSketchTable),
        bitarray=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=sketch_tables.BitarrayBackedSketchTable),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, int(n / n_factor))


def benchmark(words, sketches):
    results = OrderedDict()

    results['Insertion time'] = run_single_benchmark(
        sketches, lambda s: timeit(partial(s.update, words), number=1))

    results['Memory footprint'] = run_single_benchmark(sketches, asizeof)

    results['Depth'] = []
    results['Width'] = []
    for key in sketches:
        sk = sketches[key]
        if isinstance(sk, Counter):
            results['Depth'].append('N/A')
            results['Width'].append('N/A')

        elif isinstance(sk, double_hashing.MultiHashPairTopNCMSketch):
            results['Depth'].append(sk.sketches[0].table.depth)
            results['Width'].append(sk.sketches[0].table.width)

        elif isinstance(sk, counter_sketch_hybrid.SketchCounterHybrid):
            results['Depth'].append(sk.sketch.table.depth)
            results['Width'].append(sk.sketch.table.width)

        else:
            results['Depth'].append(sk.table.depth)
            results['Width'].append(sk.table.width)

    header = ['Metric'] + list(sketches.keys())
    print(tabulate(
        [[metric] + results[metric] for metric in results],
        header,
        numalign='decimal',
        tablefmt='fancy_grid'))


def run_single_benchmark(sketches, benchmark_func):
    return [benchmark_func(sketch) for sketch in sketches.values()]


def most_common_comparison(sketches, n):
    counter = sketches[COUNTER_KEY]
    mc = {key: value for key, value in counter.most_common(n)}
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

    percent_results = [
        [key] + [top_n[key] for top_n in top_n_percent_over.values()]
        for key in mc
    ]
    percent_results.append(['Sum Total'] + [sum(top_n.values(
    )) for top_n in top_n_percent_over.values()])
    print(tabulate(percent_results, header, tablefmt='fancy_grid'))


def test_double_hashing(words_file=DEFAULT_TEST_DATA_FILE,
                        epsilon=DEFAULT_EPSILON,
                        delta=DEFAULT_DELTA,
                        n=count_min_sketch.DEFAULT_N,
                        table_class=sketch_tables.ArrayBackedSketchTable):

    words = read_words(words_file)
    sketches = OrderedDict(
        array=count_min_sketch.TopNCountMinSketch(
            delta, epsilon, table_class=table_class),
        array_hash_pair=double_hashing.HashPairCMSketch(
            delta=n, epsilon=epsilon, table_class=table_class),
        array_hash_pair_multi=double_hashing.MultiHashPairTopNCMSketch(
            delta, epsilon, table_class=table_class),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_to_vector():
    depth = 2
    width = 10
    count = depth * width
    counts = [random.randint(1, 10) for _ in range(count)]

    vectors = []
    for table_class in (sketch_tables.ListBackedSketchTable,
                        sketch_tables.ArrayBackedSketchTable,
                        sketch_tables.NumpyMatrixBackedSketchTable,
                        sketch_tables.BitarrayBackedSketchTable):
        table = table_class(depth, width)
        for d in range(depth):
            for i in range(width):
                table.set(d, i, counts[d * width + i])

        vectors.append(table.to_vector())

    for v in vectors:
        print(v)


def test_least_squares(words_file=DEFAULT_TEST_DATA_FILE,
                       epsilon=DEFAULT_EPSILON,
                       delta=DEFAULT_DELTA,
                       n=count_min_sketch.DEFAULT_N,
                       table_class=sketch_tables.ArrayBackedSketchTable):
    words = read_words(words_file)
    sketches = OrderedDict(
        array=count_min_sketch.TopNCountMinSketch(
            delta, epsilon, table_class=table_class),
        array_hash_pair=double_hashing.HashPairCMSketch(
            delta=n, epsilon=epsilon, table_class=table_class),
        array_hash_pair_multi=double_hashing.MultiHashPairTopNCMSketch(
            delta, epsilon, table_class=table_class),
        least_squares=least_squares_sketch.LeastSquaresTopNSketch(
            delta, epsilon, table_class=table_class),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_update_strategies(words_file=DEFAULT_TEST_DATA_FILE,
                           epsilon=DEFAULT_EPSILON,
                           delta=DEFAULT_DELTA,
                           n=count_min_sketch.DEFAULT_N,
                           table_class=sketch_tables.ArrayBackedSketchTable,
                           n_factor=5):
    words = read_words(words_file)
    # Using a factor here to account for different values in the exact top n
    n *= n_factor

    sketches = OrderedDict(
        array=count_min_sketch.TopNCountMinSketch(
            delta, epsilon, n=n, table_class=table_class),
        array_hash_pair=double_hashing.HashPairCMSketch(
            delta, epsilon, n=n, table_class=table_class),
        array_conservative=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        array_hash_pair_conservative=double_hashing.HashPairCMSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        counter=Counter())

    # This test does generate the hash-pair ones and the non-hash-pair ones
    # with different sizes. However, forcing the non-hash-pair ones into the
    # hash-pair scheme only generates worse results

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_count_mean(words_file=DEFAULT_TEST_DATA_FILE,
                    epsilon=DEFAULT_EPSILON,
                    delta=DEFAULT_DELTA,
                    n=count_min_sketch.DEFAULT_N,
                    table_class=sketch_tables.ArrayBackedSketchTable,
                    n_factor=5):
    words = read_words(words_file)
    # Using a factor here to account for different values in the exact top n
    n *= n_factor

    sketches = OrderedDict(
        hash_pair=double_hashing.HashPairCMSketch(
            delta, epsilon, n=n, table_class=table_class),
        hash_pair_cons=double_hashing.HashPairCMSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        count_mean=count_mean_sketch.CountMeanMinSketch(
            delta, epsilon, n=n, table_class=table_class),
        hash_pair_count_mean=count_mean_sketch.HashPairCountMeanMinSketch(
            delta, epsilon, n=n, table_class=table_class),
        count_mean_cons=count_mean_sketch.CountMeanMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        hash_pair_count_mean_cons=count_mean_sketch.HashPairCountMeanMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)


def test_lossy_strategy(words_file=DEFAULT_TEST_DATA_FILE,
                        epsilon=DEFAULT_EPSILON,
                        delta=DEFAULT_DELTA,
                        gamma=0.01,
                        bucket_size=100,
                        n=count_min_sketch.DEFAULT_N,
                        table_class=sketch_tables.ArrayBackedSketchTable):
    words = read_words(words_file)
    # To test lossy strategies, we process the files similarly as before
    # And compare by buckets, akin to Goyal and Daumé (2010)
    counter = Counter()
    sketches = OrderedDict(
        no_lossy=count_min_sketch.TopNCountMinSketch(
            delta, epsilon, n=n, table_class=table_class),
        lossy_no_threshold=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.no_threshold_func)),
        lossy_1_threshold=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.one_threshold_func)),
        lossy_window_size=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.window_size_threshold_func)),
        lossy_sqrt_window=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.sqrt_window_size_threshold_func)),
        counter=counter)

    benchmark(words, sketches)
    benchmark_by_buckets(bucket_size, sketches)


def benchmark_by_buckets(bucket_size, sketches):
    counter = sketches[COUNTER_KEY]
    results_by_bucket = [
        {key: []
         for key in sketches if key != COUNTER_KEY}
        for _ in range(counter.most_common(1)[0][1] // bucket_size + 1)
    ]
    for item in counter:
        for key in sketches:
            if key == COUNTER_KEY:
                continue
            count = counter[item]

            results_by_bucket[count // bucket_size][key].append(
                (sketches[key].get(item) - count))

    # Turn into average
    results_table = []
    for i in range(len(results_by_bucket)):
        start = i * bucket_size
        bucket = '{start}-{end}'.format(
            start=start, end=start + bucket_size - 1)
        bucket_results = [results_by_bucket[i][key] for key in sketches
                          if key != COUNTER_KEY]

        if any(bucket_results):
            results_table.append([bucket] + [mean(br)
                                             for br in bucket_results])

    results_table.append(['Overall average'] + [mean([
        results_table[row][col] for row in range(len(results_table))
    ]) for col in range(1, len(results_table[0]))])
    header = ['Bucket'] + [key for key in sketches if key != COUNTER_KEY]
    print(tabulate(results_table, header, tablefmt='fancy_grid'))


def test_lossy_conservative_strategy(
        words_file=DEFAULT_TEST_DATA_FILE,
        epsilon=DEFAULT_EPSILON,
        delta=DEFAULT_DELTA,
        gamma=0.01,
        bucket_size=100,
        n=count_min_sketch.DEFAULT_N,
        table_class=sketch_tables.ArrayBackedSketchTable):

    # To test lossy strategies, we process the files similarly as before
    # And compare by buckets, akin to Goyal and Daumé (2010)
    words = read_words(words_file)
    counter = Counter()
    sketches = OrderedDict(
        no_lossy=count_min_sketch.TopNCountMinSketch(
            delta, epsilon, n=n, table_class=table_class),
        no_lossy_cons=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        lossy_no_threshold=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.no_threshold_func)),
        lossy_no_threshold_cons=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.no_threshold_func)),
        lossy_window_size=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.window_size_threshold_func)),
        lossy_sqrt_window_cons=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy,
            lossy_strategy=lossy_strategy.LossyUpdateStrategy(
                gamma, lossy_strategy.window_size_threshold_func)),
        counter=counter)

    benchmark(words, sketches)
    benchmark_by_buckets(bucket_size, sketches)


def test_sketch_hybrid(words_file=DEFAULT_TEST_DATA_FILE,
                       epsilon=DEFAULT_EPSILON,
                       delta=DEFAULT_DELTA,
                       n=count_min_sketch.DEFAULT_N,
                       table_class=sketch_tables.ArrayBackedSketchTable,
                       n_factor=5):
    words = read_words(words_file)
    # Using a factor here to account for different values in the exact top n
    n *= n_factor

    sketches = OrderedDict(
        array_conservative=count_min_sketch.TopNCountMinSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        array_hash_pair_conservative=double_hashing.HashPairCMSketch(
            delta,
            epsilon,
            n=n,
            table_class=table_class,
            update_strategy=update_strategy.ConservativeUpdateStrategy),
        hybrid_conservative=counter_sketch_hybrid.SketchCounterHybrid(
            count_min_sketch.TopNCountMinSketch(
                delta,
                epsilon,
                n=n,
                table_class=table_class,
                update_strategy=update_strategy.ConservativeUpdateStrategy)),
        hybrid_hash_pair_conservative=counter_sketch_hybrid.SketchCounterHybrid(
            double_hashing.HashPairCMSketch(
                delta,
                epsilon,
                n=n,
                table_class=table_class,
                update_strategy=update_strategy.ConservativeUpdateStrategy)),
        counter=Counter())

    benchmark(words, sketches)
    most_common_comparison(sketches, n)

if __name__ == '__main__':
    test_sketch_hybrid('shakespeare.txt')



