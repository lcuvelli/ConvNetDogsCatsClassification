""" Script to test several parameter values and selects the optimal """

import train_and_predict

"""Test epochs until n"""
def eopch_test(X, y, X_test, id_line, n, batch_size):
    epoch = n
    train_and_predict.training_and_testing_model(X, y, X_test, id_line, epoch, batch_size)


""" Test n values of batch size, from 2^4 to 2^n"""
def batch_size_test(X, y, X_test, id_line, n):
    batch_size = 16
    epoch = 6
    for i in range(n):
        batch_size *= 2
        train_and_predict.training_and_testing_model(X, y, X_test, id_line, epoch, batch_size)