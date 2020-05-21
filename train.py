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

def setupDataset(dataset):
    print("Loading....")
    data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(dataset)))
    random.seed(23)
    random.shuffle(imagePaths)
    count = 0
    for Images in imagePaths:
        count = count + 1
        print("[Status] {}/{}".format(len(imagePaths), count))
        image = cv2.imread(Image)
        image = cv2.resize(image, (75, 75)).flatten()
        data.append(image)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
