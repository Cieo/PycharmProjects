import numpy as np
import random


def getbigger(x):
    x[1] += 1

a = [[3,1],[2,2],[1,5]]
for i in a:
    i[1] = -1
print(a)