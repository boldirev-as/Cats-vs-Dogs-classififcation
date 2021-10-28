import os
import pickle

with open("output.pickle", 'rb') as f:
    data = pickle.load(f)
print("Data loaded")

ImagePaths = list(map(lambda x: x.strip(".jpg"), os.listdir("test")))
with open("output.csv", mode="w") as f:
    print("id,label", file=f)
    for i, num in enumerate(data):
        if num >= 0.5:
            print(f"{ImagePaths[i]},0", file=f)
        else:
            print(f"{ImagePaths[i]},1", file=f)
