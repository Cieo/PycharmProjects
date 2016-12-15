import math
import random


class Node:
    def __init__(self):
        self.x = (random.random() - 0.5) * 500
        self.y = (random.random() - 0.5) * 500
        self.child_nodes = list()
        self.time = random.random() * 1000
        self.time_gain = 0.8 + random.random() * 0.2
        self.sync_finish_time_glo = 0


# 计算两个节点之间的距离
def get_dist(node1, node2):
    return ((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2), node2


def build_topology(root, nodes):
    # 每个节点最多能够拥有的子节点数量
    max_child_node_num = 3

    queue = list()
    queue.append(root)
    nodes.remove(root)
    while len(queue) != 0:
        top = queue.pop(0)

        dist = list(map(lambda x: get_dist(top, x), nodes))
        dist.sort(key=lambda x: x[0])
        # 选择最近的N个节点作为子节点
        selected_child_nodes = dist[:math.ceil(random.random() * max_child_node_num)]
        for child_node in selected_child_nodes:
            queue.append(child_node[-1])
            top.child_nodes.append(child_node[-1])
            nodes.remove(child_node[-1])


# 将基准时间转换成节点自身的时间
def convert2selftime(global_time, node):
    return global_time / node.time_gain


def sync_time(root):
    # t2与t3之间的时间差（节点自身时间）
    reply_delay = 5
    # 广播传播速度
    spread_speed = 10000
    # 统计结果所使用的list，包含所有的节点
    nodes = list()

    queue = list()
    queue.append(root)

    while len(queue) != 0:
        top = queue.pop(0)
        nodes.append(top)
        # 对于各个子节点进行时间同步
        for child_node in top.child_nodes:
            # 计算传播时间与回复延迟（基准时间）
            spread_delay_glo = get_dist(top, child_node)[0] / spread_speed
            reply_delay_glo = reply_delay * top.time_gain
            # 更新子节点同步结束时间（基准时间）
            child_node.sync_finish_time_glo = top.sync_finish_time_glo
            child_node.sync_finish_time_glo += 2 * spread_delay_glo + reply_delay_glo

            # 计算双向发包收到的时间
            t1 = child_node.time
            t2 = top.time + convert2selftime(spread_delay_glo, top)
            t3 = top.time + convert2selftime(spread_delay_glo, top) + convert2selftime(reply_delay_glo, top)
            t4 = child_node.time + 2 * convert2selftime(spread_delay_glo, child_node) + convert2selftime(
                reply_delay_glo, child_node)

            # 更新子节点的时间（自身时间）
            child_node.time = t4
            child_node.time += ((t2 - t1) - (t4 - t3)) / 2

            queue.append(child_node)

    # 根据最后同步结束时间（基准时间）更新所有的节点自身的时间
    finish_time_global = max(list(map(lambda x: x.sync_finish_time_glo, nodes)))
    for node in nodes:
        node.time += convert2selftime(finish_time_global - node.sync_finish_time_glo, node)
        # 输出结果观察
        print(node.time)


if __name__ == "__main__":
    # 实验中节点的总个数
    num_of_nodes = 5
    nodes = list()

    # 构建实验初始条件
    for i in range(num_of_nodes):
        nodes.append(Node())
        # 输出初始值观察
        print(nodes[-1].time)
    print("-----------------")

    # 随机选取根节点
    root = random.sample(nodes, 1)[0]
    # 构建拓扑关系
    build_topology(root, nodes)
    # 进行时间同步
    sync_time(root)
