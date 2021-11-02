import os

from PIL import Image
from numpy import asarray

train_path = '../resource/testDataSet/'


def read_directory(in_directory: str, i: int) -> list:
    data = []
    out = [0]*3
    out[i] = 1
    in_directory += str(i)
    for address, dirs, files in os.walk(in_directory):
        for file in files:
            data.append(
                {"in": asarray(Image.open(address + "/" + file)).flatten().tolist(), "out": out}
            )
    return data


def get_dataset() -> list[dict]:
    dataset = []
    for i in range(3):
        dataset.extend(read_directory(train_path, i))
        print(str(i+1) + "0%")
    return dataset
