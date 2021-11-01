from src.feedforward import FeedForward





from PIL import Image
from numpy import asarray

train_path = '../resource/trainingSet/trainingSet'

image = Image.open(train_path + "sdfergthyl.jpg")

data = asarray(image)
print(data)


























dataset = [
    {"in": [0, 0], "out": [0]},
    {"in": [0, 1], "out": [0]},
    {"in": [1, 0], "out": [0]},
    {"in": [1, 1], "out": [1]}
]


print("AND")
neural = FeedForward(2, 3, 1)
res = neural.think(dataset["in"])
neural.learn(dataset, iteration_count=10000)
res = neural.think(dataset["in"])
print()
