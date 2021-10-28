import pickle
import random
import numpy as np
import os
from PIL import Image

ImagePaths = list(os.listdir("cats_vs_dogs"))
random.seed(42)
random.shuffle(ImagePaths)


data = list()
labels = np.array([])

for i, image_path in enumerate(ImagePaths):
    if i % 100 == 0:
        print(i)
    image = Image.open("cats_vs_dogs/" + image_path)
    image = image.resize((150, 150), Image.ANTIALIAS)
    data.append(np.array(image) / 255)

    label = image_path.split(os.path.sep)[-1].split(".")[0]

    if label == "cat":
        label = [1]
    else:
        label = [0]
    labels = np.append(labels, label)

data = np.array(data)
# [1] = кошка; [0] = собака

with open("Labels_150-150.pikle", 'wb') as f:
    pickle.dump(labels, f)
print("Labels seved")

with open("Data_150-150.pikle", 'wb') as f:
    pickle.dump(data, f)
print("Data seved")
