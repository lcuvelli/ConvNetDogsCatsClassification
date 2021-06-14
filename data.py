"""
Methods to plot the data and to import the images as grayscale images, resize them and stock in arrays
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from matplotlib.image import imread
#import tensorflow as tf
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from matplotlib.image import imread
import os


# Input data files are available in the "./dataset/" directory.

""" Plot the first n images from folder"""
def plot_images(n,folder):
    for i in range(n):
        # define subplot
        plt.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'cat.' + str(i + 4029) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        plt.imshow(image)
    # show the figure
    plt.show()


def create_training_data(path):
    convert = lambda category: int(category == 'dog')
    i = 0
    X = []
    y = []
    X_dogs = []
    X_cats = []
    for p in os.listdir(path):
        if i%100 == 0:
          print(i)
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        X.append(new_img_array)
        y.append(category)
        if (category == 1):
          X_dogs.append(new_img_array)
        else :
          X_cats.append(new_img_array)
        i = i + 1

    X = np.array(X).reshape(-1, 80, 80, 1)
    y = np.array(y)
    X_dogs = np.array(X_dogs)
    X_cats = np.array(X_cats)

    # Normalizing the data
    X = X/255.0
    X_dogs = X_dogs/255.0
    X_cats = X_cats / 255.0

    return [X, y, X_dogs, X_cats]




def main():

    #plot_images(3, "./dataset/test_set/all/")

    # Importing the data

    path = './dataset/training_set/all/'
    X, y, X_dogs, X_cats  = create_training_data(path)  # array of pixels


main()


