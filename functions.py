
import numpy as np
import cv2
import os
import math
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
#from pylab import *
#import matplotlib.cbook as cbook
#import random
#import time
#import matplotlib.image as mpimg
#from scipy.ndimage import filters



# CHANGE THIS TO THE PATH CONTAINING THE PROJECT ON YOUR COMPUTER #
path = "C:/Users/Clement/Desktop/final project"
###################################################################

# preparing the sets for gender recognition
def get_data_gender():

    # Down-scaling and grey-scaling of the images
    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                # male case
                if actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan':

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(grayed, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),'male'])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1
                    every = []
                    images = []

                # female case
                elif actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon':

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(grayed, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),'female'])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1
                    every = []
                    images = []

    return train,valid,test,train_images, test_images

# using an edges detector (canny)
def get_data_gender_canny():

    # Down-scaling and grey-scaling of the images
    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        gender = ''
        if directory_1 ==  os.path.join(path, 'actors'):
            gender = 'male'
        if directory_1 ==  os.path.join(path, 'actresses'):
            gender = 'female'

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                if (actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan' or actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon'):

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # detecting the edges with canny
                        edges = cv2.Canny(grayed,100,200)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),gender])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1

                    every = []
                    images = []

    return train,valid,test,train_images, test_images

# preparing the sets for gender recognition on the other actresses and actors
def get_other_data_gender():

    # Down-scaling and grey-scaling of the images
    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        gender = ''
        if directory_1 ==  os.path.join(path, 'actors'):
            gender = 'male'
        if directory_1 ==  os.path.join(path, 'actresses'):
            gender = 'female'

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                if not(actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan' or actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon'):

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(grayed, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),gender])
                        images.append(resized)

                        # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1

                    every = []
                    images = []

    return train,valid,test,train_images, test_images

# using an edges detector (canny)
def get_other_data_gender_canny():

    # Down-scaling and grey-scaling of the images
    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        gender = ''
        if directory_1 ==  os.path.join(path, 'actors'):
            gender = 'male'
        if directory_1 ==  os.path.join(path, 'actresses'):
            gender = 'female'

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                if not(actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan' or actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon'):

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # detecting the edges with canny
                        edges = cv2.Canny(grayed,100,200)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),gender])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1

                    every = []
                    images = []

    return train,valid,test,train_images, test_images

# preparing the sets for persons recognition
def get_data_person():

    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                if (actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan' or actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon'):

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))
                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(grayed, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),actor.name])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1
                    every = []
                    images = []

    return train,valid,test,train_images,test_images

# using an edges detector (canny)
def get_data_person_canny():

    directory_list1 = [a for a in os.listdir() if os.path.isdir(a)]

    directory_list2 = [os.path.join(path,filename) for filename in directory_list1 if (filename == 'actors' or filename == 'actresses')]

    every, train, valid, test = [],[],[],[]
    images, train_images, test_images = [],[],[]

    for directory_1 in directory_list2:

        file = [os.path.join(directory_1,filename) for filename in os.scandir(directory_1)]

        for directory_2 in file:

            for actor in os.scandir(directory_2):

                if (actor.name == 'Gerard_Butler' or actor.name == 'Daniel_Radcliffe' or actor.name == 'Michael_Vartan' or actor.name == 'Lorraine_Bracco' or actor.name == 'Peri_Gilpin' or actor.name == 'Angie_Harmon'):

                    for image in os.scandir(os.path.join(directory_2,actor)):

                        img = cv2.imread(os.path.join(os.path.join(directory_2,actor),image.name))

                        # gray scalling
                        grayed = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        # detecting the edges with canny
                        edges = cv2.Canny(grayed,100,200)
                        # downscalling to 32x32
                        dim = (32, 32)
                        resized = cv2.resize(edges, dim, interpolation = cv2.INTER_AREA)

                        # saving the image_path and the flatten image
                        every.append([((np.array(resized)).flatten()).tolist(),actor.name])
                        images.append(resized)

                    # initializing the training, test and validation sets
                    i=0
                    while i < int(len(every)*80/100):
                        train.append(every[i])
                        train_images.append(images[i])
                        i+=1
                    while i < len(every) - int((len(every)*20/100)/2):
                        valid.append(every[i])
                        i+=1
                    while i < len(every):
                        test.append(every[i])
                        test_images.append(images[i])
                        i+=1
                    every = []
                    images = []

    return train,valid,test,train_images,test_images

# L2 distance
def L2_distance(X, Y):
    sum = 0
    for i in range(len(X)):
        sum += math.pow(X[i] - Y[i], 2)
    dis = math.sqrt(sum)
    #print(dis)
    return dis

# Find the best K
def find_k(train_set, validation_set):
    efficiency = 0
    efficiencies = []
    k = 1
    while k<=50:
        prediction = []
        for x in validation_set:
            count = 0
            distances = []
            for y in train_set:
                distance = L2_distance(x[0],y[0])
                distances.append([distance, count])
                count+=1
            ordered_distances = sorted(distances, key=lambda x: x[0])
            #print("this ", ordered_distances)
            k_nearest = ordered_distances[:k]
            k_nearest_labels = Counter([(train_set[i[1]])[1] for i in k_nearest])
            most_comnon_label = k_nearest_labels.most_common(1)[0]
            prediction.append([x, most_comnon_label[0]])

        # verifying the efficiency of the algorithm for the current k
        #print(prediction)
        sum = 0
        for p in prediction:
            if p[0][1]==p[1]:
                sum+=1
        efficiency = sum / len(prediction)
        efficiencies.append([efficiency, k])
        print("For K = ", k, " --- Efficiency = ", efficiency)
        k+=1

    efficiencies = sorted(efficiencies, key=lambda x: x[0])
    print('\nBest result = ', efficiencies[len(efficiencies)-1][0], ', for k = ', efficiencies[len(efficiencies)-1][1], '\n')
    return efficiencies[len(efficiencies)-1][1]

# K-NN for one point
def knn(train_set, test_point, k):
    count = 0
    prediction = []
    distances = []
    for y in train_set:
        distance = L2_distance(test_point[0],y[0])
        distances.append([distance, count])
        count+=1
    ordered_distances = sorted(distances, key=lambda x: x[0])
    k_nearest = ordered_distances[:k]
    k_nearest_labels = Counter([(train_set[i[1]])[1] for i in k_nearest])
    most_comnon_label = k_nearest_labels.most_common(1)[0]
    prediction.append([test_point, most_comnon_label[0]])

    nearests = [train_set[i[1]][1] for i in k_nearest]
    indexs = [i[1] for i in k_nearest]
    return nearests, indexs, most_comnon_label[0]

# to plot images as points and link them to the face we looked for
def imscatter(x, y, index, ax, failures, fail_near):
    patches = []
    # adding the image we looked for to the graph
    im = OffsetImage(failures[index][0], zoom=0.7, cmap='gray')
    ab = AnnotationBbox(im, (2, 3), xycoords='data', frameon=False)
    ax.add_artist(ab)

    # adding nearest image to the graph
    for i in range(5):
        im = OffsetImage(fail_near[index][i], zoom=0.7, cmap='gray')
        ab = AnnotationBbox(im, (i, 1), xycoords='data', frameon=False)
        ax.plot([2, i], [3, 1])
        ax.add_artist(ab)

    ax.set_title(str(failures[index][1] + ': found ' + failures[index][2]))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
