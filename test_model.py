import os
import pickle
import random

from keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("point.h5")

ImagePaths = list(os.listdir("test"))
data = list()
print(len(ImagePaths))

for i, imagepath in enumerate(ImagePaths):
    if i % 100 == 0:
        print(i)
    image = Image.open("test/" + imagepath)
    image = image.resize((150, 150), Image.ANTIALIAS)
    data.append(np.reshape(np.array(image) / 255, (150, 150, 3)))
    # data = np.array(data, dtype="float") / 255.0

pred = model.predict(np.array(data))
with open("output.pickle", mode="wb") as f:
    pickle.dump(pred, file=f)
print(pred)
