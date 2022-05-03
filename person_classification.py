import functions as fn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import gc

# PART 2
train_set, validation_set, test_set, train_images, test_images = fn.get_data_person()

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
for i in range(len(predictions)):
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

# PART 4
# Ploting performances
test_performance = []
train_performance = []
val_performance = []
for k in range(10):
    predictions_for_test = []
    predictions_for_train = []
    predictions_for_val = []
    for test_point in test_set:
        k_nearest, indexs, prediction = fn.knn(train_set, test_point, k+1)
        predictions_for_test.append(prediction)
    for train_point in train_set:
        k_nearest, indexs, prediction = fn.knn(train_set, train_point, k+1)
        predictions_for_train.append(prediction)
    for val_point in validation_set:
        k_nearest, indexs, prediction = fn.knn(train_set, val_point, k+1)
        predictions_for_val.append(prediction)

    # Performances calculation
    sum_test = 0
    sum_train = 0
    sum_val = 0
    for i in range(len(test_set)):
        if test_set[i][1]==predictions_for_test[i]:
            sum_test+=1
    for i in range(len(train_set)):
        if train_set[i][1]==predictions_for_train[i]:
            sum_train+=1
    for i in range(len(validation_set)):
        if validation_set[i][1]==predictions_for_val[i]:
            sum_val+=1
    test_performance.append(sum / len(predictions_for_test))
    train_performance.append(sum / len(predictions_for_train))
    val_performance.append(sum / len(predictions_for_val))
    print('\nPerformance for K = ', k+1, ' on the test set is -->', test_performance[k])
    print('Performance for K = ', k+1, ' on the train set is -->', train_performance[k])
    print('Performance for K = ', k+1, ' on the val set is -->', val_performance[k], '\n')

K = [i+1 for i in range(10)]

plt.plot(K,test_performance, label='test_performance', color='c')
plt.plot(K,train_performance, label='train_performance', color='r')
plt.plot(K,val_performance, label='val_performance', color='g')

plt.legend(loc='lower left')
plt.xlabel('k')
plt.ylabel('performance')

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
del test_performance
del train_performance
del val_performance
gc.collect()
