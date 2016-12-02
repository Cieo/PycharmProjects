import random
import numpy as np

def get_dist(x1, x2):
    return sum(list(map(lambda x: (x1[x] - x2[x]) ** 2, range(len(x1) - 1))))

a = [1,2,2]
b = [2,1,3]

print(get_dist(a,b))