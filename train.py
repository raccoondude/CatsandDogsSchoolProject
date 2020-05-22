import matplotlib
from sklearn.preprocessing import LabelBinarizer
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

E = "10"

def setupDataset(dataset):
    print("Loading....")
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(42)
    random.shuffle(imagePaths)
    count = 0
    for Image in imagePaths:
        count = count + 1
        print("[Status] {}/{}".format(len(imagePaths), count))
        image = cv2.imread(Image)
        image = cv2.resize(image, (75, 75)).flatten()
        data.append(image)
        label = Image.split(os.path.sep)[-2]
        labels.append(label)
    return data,labels

def makeTree():
    tree = tf.keras.Sequential()
    tree.add(tf.keras.layers.Dense(1024, input_shape=(16875,), activation="sigmoid"))
    tree.add(tf.keras.layers.Dense(700, activation="relu"))
    #tree.add(tf.keras.layers.Dense(500, activation="relu"))
    tree.add(tf.keras.layers.Dense(200, activation="relu"))
    #tree.add(tf.keras.layers.Dense(100, activation="relu"))
    tree.add(tf.keras.layers.Dense(50, activation="relu"))
    tree.add(tf.keras.layers.Dense(10, activation="relu"))
    tree.add(tf.keras.layers.Dense(len(lb.classes_), activation="softmax"))
    tree.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return tree

data, labels = setupDataset("training_set")
print("[Status] Finished!")

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[System] Building tree")
tree = makeTree()

print("[Training] Training has started")
Train = tree.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=32)
print("[Training] Training has ended")


print("[Status] Evaluating.....")

predictions = tree.predict(testX, batch_size=32)

N = np.arange(0, 10)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, Train.history["loss"], label="train_loss")
plt.plot(N, Train.history["val_loss"], label="val_loss")
plt.plot(N, Train.history["accuracy"], label="train_acc")
plt.plot(N, Train.history["val_accuracy"], label="accuracy")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(E+".png")

print("[Status] Saving")
plt.savefig(E+".png")
f = open(E+".h5.bin", "wb")
f.write(pickle.dumps(lb))
f.close()
