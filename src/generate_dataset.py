import os
from PIL import Image
from numpy import asarray


def read_dataset_in_path(path: str, number: int) -> list:
    data = []
    out = [0, 0, 0]
    out[number] = 1
    for address, dirs, files in os.walk(path + str(number)):
        for file in files:
            data.append(
                {"in": [x / 255 for x in asarray(Image.open(address + "/" + file)).flatten().tolist()], "out": out}
            )
    return data


def get_in_list(address) -> list:
    l255 = asarray(Image.open(address)).flatten().tolist()
    return [x / 255 for x in l255]

# def read_directory_bytes(in_directory: str, i: int) -> list:
#     data = []
#     out = [0]*3
#     out[i] = 1
#     in_directory += str(i)
#     for address, dirs, files in os.walk(in_directory):
#         for file in files:
#             with open(address + "/" + file, "rb") as fil:
#                 l255 = list(fil.read())
#             new_l = [x / 255 for x in l255]
#             data.append(
#                 {"in": new_l, "out": out}
#             )
#     return data


def read_dataset(train_path) -> list[dict]:
    dataset = []
    for i in range(3):
        dataset.extend(read_dataset_in_path(train_path, i))
    return dataset


# def get_dataset2() -> list[dict]:
#     dataset = []
#     for i in range(3):
#         dataset.extend(read_directory_bytes(train_path2, i))
#         print(str(i+1) + "0%")
#     return dataset