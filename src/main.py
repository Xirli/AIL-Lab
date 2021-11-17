from src.feedforward import FeedForward
from src.generate_dataset import read_dataset, get_in_list


def to_str(l: list) -> str:
    return " ".join("{:.3f}".format(x) for x in l)


dataset = read_dataset('../resource/testDataSet/')
neural = FeedForward(784, 10, 5, 3)


result = neural.think(dataset[0]["in"])
print(to_str(result) + " | " + to_str(dataset[0]["out"]))

result = neural.think(dataset[10]["in"])
print(to_str(result) + " | " + to_str(dataset[10]["out"]))

result = neural.think(dataset[20]["in"])
print(to_str(result) + " | " + to_str(dataset[20]["out"]))


neural.learn(dataset, iteration_count=100)
print()


result = neural.think(dataset[0]["in"])
print(to_str(result) + " | " + to_str(dataset[0]["out"]))

result = neural.think(dataset[10]["in"])
print(to_str(result) + " | " + to_str(dataset[10]["out"]))

result = neural.think(dataset[20]["in"])
print(to_str(result) + " | " + to_str(dataset[20]["out"]))



a = get_in_list("../resource/testSet/img_2.jpg")
result = neural.think(a)
print(to_str(result))

# a = get_in_list("../resource/myNumber/img_1.jpg")
# print(len(a))
# result = neural.think(a)
# print(to_str(result))
