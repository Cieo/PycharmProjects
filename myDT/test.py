import numpy as np

def getID3(attr, train_data, entropy):
    classes = set(map(lambda x: x[attr], train_data))
    size = len(train_data)
    entropy_attr = 0
    for clas in classes:
        class_size = len(list(filter(lambda x: x[attr] == clas, train_data)))
        class_right = len(list(filter(lambda x: x[attr] == clas and x[-1] == 1, train_data)))
        class_error = class_size - class_right
        if class_right != 0 and class_error != 0:
            entropy_attr += class_size / size * (
                -class_right / class_size * np.log2(class_right / class_size) - class_error / class_size * np.log2(
                    class_error / class_size))
        elif class_error == 0:
            entropy_attr += class_size / size * (-class_right / class_size * np.log2(class_right / class_size))
        elif class_right == 0:
            entropy_attr += class_size / size * (-class_error / class_size * np.log2(class_error / class_size))
    return attr, entropy - entropy_attr


def bulid_tree(root, attr, train_data):
    size = len(train_data)
    right = len(list(filter(lambda x: x[-1] == 1, train_data)))
    error = size - right
    if size == right:
        root["result"] = 1
        return
    elif size == error:
        root["result"] = 0
        return
    elif len(attr) == 0:
        if right > error:
            root["result"] = 1
        else:
            root["result"] = 0

    entropy = 0
    if right == 0:
        entropy = -error / size * np.log2(error / size)
    elif error == 0:
        entropy = -right / size * np.log2(right / size)
    elif right != 0 and error != 0:
        entropy = -right / size * np.log2(right / size) - error / size * np.log2(error / size)
    best_attr = (sorted(list(map(lambda x: getID3(x, train_data, entropy), attr)), key=lambda x: x[-1])[-1])[0]
    classes = set(map(lambda x: x[best_attr], train_data))
    attr_copy = list(attr)
    attr_copy.remove(best_attr)
    for clas in classes:
        root[clas] = dict()
        bulid_tree(root[clas], attr_copy, list(filter(lambda x: x[best_attr] == clas, train_data)))

if __name__ == "__main__":
    root = dict()
    a = list()
    a.append(['a','c',1])
    a.append(['a','d',0])
    a.append(['b','c',0])
    a.append(['b','d',1])
    bulid_tree(root,[0,1],a)
    print(root)
    print(round(int(15)/10))