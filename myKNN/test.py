import numpy as np
import random

emotions = {"anger": 3, "disgust": 0, "fear": 0, "guilt": 0, "joy": 0, "sadness": 0, "shame": 0}
predict = sorted(emotions.items(), key=lambda x: x[1])[-1][0]
print(predict)