import numpy as np
from scipy.stats import entropy
from collections import Counter
from time import  clock
import random
from Mutual_Information import binary_target_helper
import torch
random.seed(1)

test_array = torch.randint(0,9,(60000,), dtype=torch.int64)
# test_array = [random.randint(0,9) for x in range(10000)]
print(type(test_array).__name__)
time_start = clock()
# index_set = {5,6,7}
# index_arr = [5,6,7]
test_array = binary_target_helper(test_array, [5,6,7,8,9,0])
# test_array = [index_arr.index(x)+1 if x in index_set else 0 for x in test_array]
time_end = clock()

print(test_array, "\n", time_end-time_start)

#
# counts = list(Counter(z).values())
#
# print(counts)
#
# print(entropy(list(map(lambda x: x/len(z), counts))))