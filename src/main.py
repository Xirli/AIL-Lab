from math import ceil

import numpy as np

from src.feedforward import FeedForward

from src.generate_dataset import get_dataset


def list_str(l: list) -> str:
    return " ".join(str(ceil(x - 0.5)) for x in l)


def list_str2(l: list) -> str:
    return str(np.argmax(np.array(l)))


def list_str3(l: list) -> str:
    return " ".join("{:.20f}".format(x) for x in l)


dataset = get_dataset()

neural = FeedForward(784, 1, 3)

result = neural.think(dataset[0]["in"])
print(result)

neural.learn(dataset, iteration_count=10)

result = neural.think(dataset[0]["in"])
print(list_str3(result[0:3]) + " | " + list_str(dataset[0]["out"]))

result = neural.think(dataset[10]["in"])
print(list_str3(result[0:3]) + " | " + list_str(dataset[10]["out"]))

result = neural.think(dataset[20]["in"])
print(list_str3(result[0:3]) + " | " + list_str(dataset[20]["out"]))
