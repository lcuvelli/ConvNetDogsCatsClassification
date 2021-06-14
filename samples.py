"""

From the dataset, selects randomly 2n images with n images of dogs and n images of cats
and keeps the corresponding labels

"""
import random
import numpy as np # linear algebra


def select_random_images(n,X,y):
    randomlist = random.sample(range(0, 7999), 3000)
    print(randomlist)

    X_al = []
    y_al = []

    for i in range(len(randomlist)):
        X_al.append(X[randomlist[i]])
        y_al.append(y[randomlist[i]])

    X_al = np.array(X_al).reshape(-1, 80, 80, 1)
    y_al = np.array(y_al)

    return [X_al, y_al]