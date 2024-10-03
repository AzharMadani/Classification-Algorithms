# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob

# Reading a single image

image = cv2.imread('a.jpeg')
image.shape
# (1109, 1616, 3) ----> (Width, Height, Encoding {RGB})
# Total no of pixels in a.jpeg = 1109 * 1616 * 3 = 53,76,432

# Reading the entire folder in one go

normal = [cv2.imread(file) for file in glob.glob('Normal Patients/*.jpeg')]
covid = [cv2.imread(file) for file in glob.glob('COVID Patients/*.jpeg')]

cv2.imshow('first', normal[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('first', covid[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image processing -> resize all the images into the same
# shape (dimension)
# The agreed upon dimension would be (300, 300, 3)

covid_data = [cv2.resize(image, (300, 300)) for image in covid]
normal_data = [cv2.resize(image, (300, 300)) for image in normal]

cv2.imshow('first', normal_data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('first', covid_data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Type Casting the list to numpy arrays

normal_data = np.array(normal_data)
covid_data = np.array(covid_data)

normal_data.shape
# (100, 300, 300, 3) -> 100 images of 300 * 300 width & height
# and 3 encoding channels
# (100, 300, 300, 3) -> Flattening -> (100, 300 * 300 * 3)
# Output --> (100, 270000) --> 2 Dimensional

# Combining the datasets together and creating a feature matrix

X = np.concatenate([covid_data, normal_data])

# Label generation
# Defining the rules
# 0 --> Normal Chest X-Ray
# 1 --> Covid +ve Chest X-Ray

normal_label = np.zeros(len(normal_data))
covid_label = np.ones(len(covid_data))

y = np.concatenate([covid_label, normal_label])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Giving target variable the english names

y_train_names = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_train]

# EDA

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i])
    plt.xlabel(y_train_names[i])
plt.tight_layout()
plt.show()

# Flattening the images

X_train_f = X_train.reshape(101, 270000)
X_test_f = X_test.reshape(34, 270000)

# Implementing the ML model

import lightgbm as lgb
from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier()
lgb_clf.fit(X_train_f, y_train)

lgb_clf.score(X_train_f, y_train) #1.0
lgb_clf.score(X_test_f, y_test)  #0.79

y_pred = lgb_clf.predict(X_test_f)
y_pred_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_pred]
y_test_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_test]

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test_f[i].reshape(300, 300, 3))
    plt.xlabel('Actual Label: {}\nPrediction: {}'.format(y_test_name[i], y_pred_name[i]))
plt.tight_layout()
plt.show()

# Task: Write an automation function that receives the path of all the images stored, 
# the resizing dimension, and the algorithm to eventually output accuracy and a figure 
# for actual and predicted label.

def covid_ml_module(path, dimension, estimator):
    normal = [cv2.imread(file) for file in glob.glob(path+'Normal Patients/*.jpeg')]
    covid = [cv2.imread(file) for file in glob.glob(path+'COVID Patients/*.jpeg')]
        
    covid_data = [cv2.resize(image, dimension) for image in covid]
    normal_data = [cv2.resize(image, dimension) for image in normal]

    normal_data = np.array(normal_data)
    covid_data = np.array(covid_data)
    
    X = np.concatenate([covid_data, normal_data])
    normal_label = np.zeros(len(normal_data))
    covid_label = np.ones(len(covid_data))
    y = np.concatenate([covid_label, normal_label])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_train_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_train]
    
    no_of_features = dimension[0] * dimension[1] * 3
    
    X_train_f = X_train.reshape(len(y_train), no_of_features)
    X_test_f = X_test.reshape(len(y_test), no_of_features)
    
    estimator.fit(X_train_f, y_train)
    print("**************************************************")
    print("________"+estimator.__class__.__name__+"__________")
    print()
    print("Train Accuracy: {}".format(estimator.score(X_train_f, y_train)))
    print("Test Accuracy: {}".format(estimator.score(X_test_f, y_test) ))
    print()
    print("**************************************************")
    y_pred = estimator.predict(X_test_f)
    y_pred_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_pred]
    y_test_name = ['Covid +ve' if i == 1.0 else 'Normal' for i in y_test]

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(X_test_f[i].reshape(dimension[0], dimension[1], 3))
        plt.xlabel('Actual Label: {}\nPrediction: {}'.format(y_test_name[i], y_pred_name[i]))
    plt.tight_layout()
    plt.show()

path = r"C:\Users\azhar\OneDrive\Desktop\Desktop"
dimension = (100, 100)
covid_ml_module(path, dimension, LGBMClassifier())  













