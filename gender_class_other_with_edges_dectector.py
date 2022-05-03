import functions as fn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gc

train_set, validation_set, test_set, train_images, test_images = fn.get_other_data_gender_canny()

k = fn.find_k(train_set, validation_set)

predictions = []
nearests = []
for test_point in test_set:
    k_nearest, indexs, prediction = fn.knn(train_set, test_point, k)
    predictions.append(prediction)
    nearests.append(k_nearest)

for i in range(len(nearests)):
    print('Prediction for --> ', test_set[i][1], ' --> is --> ', predictions[i], " --> from --> ", nearests[i])

# Measuring the performance
sum = 0
for i in range(len(test_set)):
    if test_set[i][1]==predictions[i]:
        sum+=1
performance = sum / len(predictions)
print('\nPerformance for K = ', k, ' on the test set is -->', performance, '\n')

del train_set
del validation_set
del test_set
del train_images
del test_images
del predictions
del nearests
gc.collect()
