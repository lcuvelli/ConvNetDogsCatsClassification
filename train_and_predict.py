import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from matplotlib.image import imread
import data

""" Train the model et make predictions on the testing set """


def training_and_testing_model(X, y, X_test, id_line, epoch, batch_size):
    model = Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add another:
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam",
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X, y, epochs=epoch, batch_size=batch_size, validation_split=0.2)


    predictions = model.predict(X_test)
    predicted_val = [int(round(p[0])) for p in predictions]
    submission_df = pd.DataFrame({'id': id_line, 'label': predicted_val})
    submission_df.to_csv("submission.csv", index=False)
    true_label = []
    for i in range(len(submission_df)):
        if (submission_df['id'][i] == 'dog'):
            true_label.append(1)
        else:
            true_label.append(0)
    submission_df['true_label'] = true_label

    # print(len(submission_df))
    error = 0
    for i in range(len(submission_df)):
        if (submission_df['label'][i] != submission_df['true_label'][i]):
            error += 1
    print(error)
    print("% of error is :", error / len(submission_df) * 100, "%")




def main():
    epoch = 6
    batch_size = 32
    train_dir = './dataset/training_set/all/'
    test_dir = './dataset/test_set/all/'

    X_test, id_line = data.create_test_data(test_dir)
    X, y, X_dogs, X_cats = data.create_training_data(train_dir)
    training_and_testing_model(X, y, X_test, id_line, epoch, batch_size)

