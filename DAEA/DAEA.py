import random
import math


class Node:
    def __init__(self, row = -1, colume = -1):
        self.x = -1
        self.y = -1
        self.energy = 100
        self.times_of_head = 0


class Cluster:
    def __init__(self):
        self.head = Node()
        self.nodes = list()

    def get_head(self):
        result = list(map(lambda x: get_weight(x), self.nodes))
        result.sort(key=lambda x: x[0])
        self.head = result[-1]
        self.head.times_of_head += 1


def get_weight(node):
    a = 0.1
    b = 1 - a
    return node.energy * a + math.exp(node.times_of_head) * b, node


def get_head_for_clusters(clusters):
    for cluster in clusters:
        cluster.get_head()


def get_ma(clusters, num_of_ma):
    pass

if __name__ == "__main__":
    random.seed(1)

    num_of_ma = 3
    num_of_cluster = 3
    max_nodes_in_cluster = 5
    period_of_change_head = 100
    times_of_period = 10

    clusters = list()
    for i in range(num_of_cluster):
        single_cluster = Cluster()
        for j in range(math.ceil(random.random() * max_nodes_in_cluster)):
            single_cluster.nodes.append(Node())
        clusters.append(single_cluster)
