import functions as fn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gc

# PART 2
train_set, validation_set, test_set, train_images, test_images = fn.get_data_gender()

# PART 3
k = fn.find_k(train_set, validation_set)

predictions = []
nearests = []
indexes = []
for test_point in test_set:
    k_nearest, indexs, prediction = fn.knn(train_set, test_point, k)
    predictions.append(prediction)
    nearests.append(k_nearest)
    indexes.append(indexs)

for i in range(len(nearests)):
    print('Prediction for --> ', test_set[i][1], ' --> is --> ', predictions[i], " --> from --> ", nearests[i])

# Measuring the performance
sum = 0
failures = [] # saving failures
fail_near = [] # saving failures' k nearest neighbours
for i in range(len(test_set)):
    if test_set[i][1]==predictions[i]:
        sum+=1
    else:
        failures.append([test_images[i], test_set[i][1], predictions[i]])
        fail_near.append([train_images[j] for j in indexes[i]])
performance = sum / len(predictions)
print('\nPerformance for K = ', k, ' on the test set is -->', performance, '\n')

# Displaying 6 failures cases with their 5 nearest neighbors
x = np.array([0,1,2,3,4])
y = np.array([0,1,2,3,4])
fig, ax = plt.subplots(nrows=2, ncols=3)
for i in range(2):
    for j in range(3):
        if i == 0:
            fn.imscatter(x, y, j, ax[i,j], failures, fail_near)
        if i == 1:
            fn.imscatter(x, y, j+3, ax[i,j], failures, fail_near)
plt.show()

del train_set
del validation_set
del test_set
del train_images
del test_images
del predictions
del nearests
del indexes
del failures
del fail_near
gc.collect()
