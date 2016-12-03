import random
import numpy as np

def get_dist(x1, x2):
    return sum(list(map(lambda x: (x1[x] - x2[x]) ** 2, range(len(x1) - 1))))


def get_choice(center, line):
    result = list(map(lambda x: get_dist(x, line), center))
    min_value = min(result)
    return result.index(min_value)


def get_cluster(train_data, k):
    center_old = random.sample(train_data, k)
    center = random.sample(train_data, k)
    while center != center_old:
        cluster = [list() for x in range(k)]
        center_old = center
        for line in train_data:
            cluster[get_choice(center, line)].append(line)
        center = list(map(lambda x: list(sum(np.array(x)) / len(x)), cluster))
        print("center", center)
        print("centol", center_old)
    return center, cluster

if __name__ == "__main__":
    a = list()
    a.append([0,2,3])
    a.append([0,0,3])
    a.append([1.5,0,3])
    a.append([5,0,3])
    a.append([5,2,3])
    get_cluster(a,2)
