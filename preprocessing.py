import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from pickle import dump
import matplotlib.pyplot as plt

BASE_DIR = "./data"
CLASSES = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

data = list()
for items in tqdm(CLASSES):
    path = os.path.join(BASE_DIR, items)
    label = CLASSES.index(items)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (80, 80))
        data.append([img_arr, label])

print("The length of data is:", len(data))

# Data Preprocessing
random.shuffle(data)

x, y = list(), list()
for features, labels in data:
    x.append(features)
    y.append(labels)

# Converting lists x and y to numpy arrays
x = np.array(x)
y = np.array(y)

print("The shape of the data features is:", x.shape)
print("The shape of the data class labels is:", y.shape)

dump(x, open('features.pkl', 'wb'))
dump(y, open('labels.pkl', 'wb'))