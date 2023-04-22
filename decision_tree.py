#Second Solution - Peak Classifier
#Decision Tree

#Import all the libraries
#May need to download sklearn.tree
import scipy.io as spio
from scipy import fftpack
import matplotlib.pyplot as plt
import numpy as np
import TimeFreq as TF
import F1Calculator as fcal
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Shift labels from 1-5 to 0-4
def correct_labels(targets):
    labels = []
    for item in targets:
        label = item - 1
        labels.append(label)
    return labels


#Train and output the decision tree model.
def output_tree_model(dataset, indexes, classes):
    class_num = 5 # Define class size

    #Convert time series data into frequency series samples of the dataset 
    unsorted_x = TF.time_freq(dataset, indexes)


    unsorted_y = correct_labels(classes) # Shift labels from 1-5 to 0-4
    y, X = zip(*sorted(zip(unsorted_y, unsorted_x))) # Sort the data by the labels in ascending order.

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(list(X), list(y), test_size=test_size, random_state=seed) # Split the data into training and test data.

    model = DecisionTreeClassifier() # Create a decision tree classifier object.
    model = model.fit(X_train,y_train) # Train the classifier model.

    y_pred = model.predict(X_test) #Predict the response for test dataset

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) # Display the accuray.
    fcal.calculate_multi_F1(y_test, y_pred, class_num) # Display the F1 score
    return model



mat = spio.loadmat('training.mat')
d = mat['d']
Index = mat['Index']
Class = mat['Class']

model = output_tree_model(d[0], Index[0], Class[0])
