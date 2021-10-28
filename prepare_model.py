import matplotlib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras.callbacks import ModelCheckpoint


with open("Data_150-150.pikle", 'rb') as f:
    data = pickle.load(f)
print("Data loaded")
with open("Labels_150-150.pikle", 'rb') as f:
    labels = pickle.load(f)
print("Labels loaded")
# Image.fromarray((data[-1] * 255).astype(np.uint8)).show()
# print(labels[-1])


trainx, testx, trainy, testy = train_test_split(
    data, labels, test_size=0.15, random_state=42)
print("ok")


model = Sequential()
model.add(Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
# model.add(Dense(1024,input_shape=(3072,), activation='sigmoid'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# INIT_LR = 0.01
# opt = SGD(lr=INIT_LR)
# opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(learning_rate=1e-4),
              metrics=["accuracy"])  # categorial_crosentropy
print("Model compiled")
model.summary()


EPOCHS = 70
# early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
checkpointer = ModelCheckpoint(
    filepath="point.h5",
    verbose=1, save_best_only=True)
H = model.fit(trainx, trainy, validation_data=(testx, testy),
              epochs=EPOCHS, batch_size=64,
              callbacks=[checkpointer])


matplotlib.use("Agg")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Results")
plt.xlabel("Epochn #")
plt.ylabel("Loss/Accuracy/")
plt.legend()
plt.savefig("Loss.png")

model.save("EasyNet.model")

print("End")
