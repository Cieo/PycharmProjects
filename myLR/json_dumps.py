import json
import numpy as np


def read_train(path):
    with open(path) as f:
        lines = f.readlines()
        lines.remove(lines[0])
        data = list()
        max_value = [-9999] * 9
        min_value = [9999] * 9
        for i in range(len(lines)):
            splits = list()
            splits.extend(lines[i].replace('\n', '').split(','))
            splits = list(map(lambda x: float(x), splits))
            for j in range(len(splits) - 1):
                max_value[j] = max(splits[j], max_value[j])
                min_value[j] = min(splits[j], min_value[j])
            data.append(splits)
        for i in range(len(lines)):
            for j in range((np.shape(data)[1]) - 1):
                (data[i])[j] = ((data[i])[j] - min_value[j]) / (max_value[j] - min_value[j])
        with open("train.json", "wt") as wf:
            json.dump(data, wf, indent=4)


if __name__ == "__main__":
    read_train("train.csv")
