# minsketch

A flexible implementation of several min-sketch variants.

A count-min sketch is a probabilistic counting data structure - to save
space, we allow for some known probability of counting error. This 
allows to summarize a data stream with limited space, without growing
indefinitely as we see more and more data.

For a longer description, and some of the relevant papers, see:
https://sites.google.com/site/countminsketch/

Here's an example usage:

```python
import minsketch
from functools import partial
from timeit import timeit

# Delta and epsilon and uncertainty measures
# Count-min sketches are within epsilon of the true value 
# With probability 1 - delta
delta = 10 ** -5
epsilon = 0.001

# This implementation supports several different backing table classes
# For most purposes, the Python array makes for a good default:
table_class = minsketch.sketch_tables.ArrayBackedSketchTable

# If you are willing to only support positive updates (no deletions)
# You should use conservative updates, as significantly reduces error:
update_strategy = minsketch.update_strategy.ConservativeUpdateStrategy

# If you have a reason to believe you might want lossy updating, you
# should benchmark one of the lossy updating schemes on your data:
lossy_strategy = minsketch.lossy_strategy.NoLossyUpdateStrategy
 
# So far, the hash-pair based implementation appears to outperform
# the universal hash family based one:
sketch = minsketch.double_hashing.HashPairCMSketch(
    delta, epsilon, table_class=table_class,
    update_strategy=update_strategy, lossy_strategy=lossy_strategy)
    
# Update the sketch with some data
from numpy import random
data = random.randint(0, 1000, 100000)
print(timeit(partial(sketch.update, data), number=1))

# Query the ten most common elements:
print(sketch.most_common(10))

# For a performance boost, you can also use a Counter-Sketch Hybrid
hybrid = minsketch.counter_sketch_hybrid.SketchCounterHybrid(
    minsketch.double_hashing.HashPairCMSketch(
    delta, epsilon, table_class=table_class,
    update_strategy=update_strategy, lossy_strategy=lossy_strategy))

print(timeit(partial(hybrid.update, data), number=1))
print(hybrid.most_common(10))
```

# Documentation

See the documentation at http://minsketch.readthedocs.io/en/latest/
