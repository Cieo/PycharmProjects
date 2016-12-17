import random
import math


class Node:
    def __init__(self, init_energy):
        self.x = (random.random() - 0.5) * 100
        self.y = (random.random() - 0.5) * 100
        self.energy = init_energy
        self.head_round = -9999
        self.next_jump = -1


class Cluster:
    def __init__(self, head):
        self.head = head
        self.nodes = list()


# 计算两个节点之间的距离
def get_dist(node1, node2):
    return (node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2


# 选择离当前节点最近的簇
def get_choice(node, clusters):
    dist = list(map(lambda x: get_dist(node, x.head), clusters))
    return dist.index(min(dist))


# 对传感器网络节点进行分簇
def get_cluster(p, nodes, current_round, head_num):
    clusters = list()
    threshold = p / (1 - p * (current_round % (1 / p)))
    # 选择簇头
    while True:
        for node in nodes:
            n = random.random()
            if n < threshold and (current_round - node.head_round) >= math.floor(1 / p) and node.energy > 0:
                clusters.append(Cluster(node))
                node.head_round = current_round
            # 选出要求数量的簇头，结束选择
            if len(clusters) == head_num:
                break
        # 如果没有选出簇头，重新选择
        if len(clusters) > 0:
            break

    # 对其余节点分簇
    for node in nodes:
        if len(clusters) > 0:
            clusters[get_choice(node, clusters)].nodes.append(node)
    return clusters


# 使用LEACH进行信息传播
def transport_data(clusters, sink, k):
    for cluster in clusters:
        for node in cluster.nodes:
            node.energy -= (get_dist(node, cluster.head) ** 2) * k
        cluster.head.energy -= (get_dist(cluster.head, sink) ** 2) * k


# 统计网络剩余能量
def get_total_energy(clusters, init_energy):
    total = 0
    for cluster in clusters:
        total += sum(list(map(lambda x: x.energy if x.energy > 0 else 0, cluster.nodes)))
    print("Total energy left", total / init_energy)
    return total


# 统计生存节点数
def get_live_nodes(clusters, init_nodes):
    total = 0
    for cluster in clusters:
        total += sum(list(map(lambda x: 1 if x.energy > 0 else 0, cluster.nodes)))
    print("Total node left", total / init_nodes)
    return total


# 生成EE-LEACH的最优路径
def get_best_route(clusters, sink):
    heads = list(map(lambda x: x.head, clusters))
    next_jump = sink
    while len(heads) > 0:
        result = list(map(lambda x: get_dist(x, next_jump), heads))
        target = heads[result.index(min(result))]
        heads.remove(target)
        target.next_jump = next_jump
        next_jump = target


# 使用EE-LEACH的最优路径进行信息传播
def transport_data_with_best_route(clusters, sink, k):
    for cluster in clusters:
        for node in cluster.nodes:
            node.energy -= (get_dist(node, cluster.head) ** 2) * k
        cluster.head.energy -= (get_dist(cluster.head, cluster.head.next_jump) ** 2) * k


if __name__ == "__main__":
    # 仿真用参数
    round_num = 100
    round_period = 100
    node_num = 200
    head_num = 31
    init_energy = 100
    p = head_num / node_num

    # 生成仿真用的节点
    sink = Node(init_energy)
    nodes = list()
    for i in range(node_num):
        nodes.append(Node(init_energy))

    # 进行普通的LEACH的仿真
    clusters = list()
    for current_round in range(round_num):
        for current_time in range(round_period):
            if current_time == 0:
                clusters = get_cluster(p, nodes, current_round, head_num)
            else:
                transport_data(clusters, sink, 0.000000001)
        print(current_round, "normal---------------------")
        get_total_energy(clusters, node_num * init_energy)
        get_live_nodes(clusters, node_num)

    # 恢复仿真用的节点的状态
    for node in nodes:
        node.energy = 100
        node.head_round = -9999

    # 进行EE的LEACH的仿真
    clusters = list()
    for current_round in range(round_num):
        for current_time in range(round_period):
            if current_time == 0:
                clusters = get_cluster(p, nodes, current_round, head_num)
                get_best_route(clusters, sink)
            else:
                transport_data_with_best_route(clusters, sink, 0.000000001)
        print(current_round, "best---------------------")
        get_total_energy(clusters, node_num * init_energy)
        get_live_nodes(clusters, node_num)
