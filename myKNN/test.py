import numpy as np
import random

with open("./stop_word.txt") as f:
    lines = set(f.read().split())
    print(lines)
