#First Solution
#Use the xgboost library for both peak detection and classification

#Libraries to be downloaded:
# xgboost


import scipy.io as spio
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import F1Calculator as fcal
import matplotlib.pyplot as plt
from scipy import fftpack


#Separate the dataset into 20x1 samples.
#Can be used only for peak classification.
#Can be used for peak detection but it will take hours with complete dataset.
def process_positions_classify(dataset, positions):
    output_range = 10 #Choose the size of the sample.
    classification_input = []
    #Loop thorugh every peak in the postions.
    for peak in positions:
        if peak > output_range  and peak < len(dataset) - output_range:
            data_list = list(dataset[peak-output_range:peak+output_range]) # Take the sample around the peak.
            classification_input.append(data_list) # Add the sample to the list.
    return classification_input


#Separate the dataset into 20x1 samples.
#Can be used only for peak detection, or both classification and detection together.
def process_positions_detect(dataset):
    output_range = 10 #Choose the size of the sample.
    classification_input = []
    data_list = list(dataset[0:output_range*2]) #Take the first sample in the dataset.
    #Loop thorugh every point in the dataset.
    for i in range (0,len(dataset)):
        #When you can begin to take a sample around the central point:
        if i > output_range  and i < len(dataset) - output_range:
            data_list = data_list[1:] # remove the first datapoint in the sample
            data_list.append(dataset[i + output_range]) # add the new datapoint to the sample.
        # This is not in the if loop as the samples need to be in the correct position.
        classification_input.append(data_list) # add the sample to the list. 
    return classification_input


#For peak detection and classification.
#Will generate labels of 0(not peak) and 1 to 5 for each type of peak 
def generate_labels_classify(Index, Class, length):
    labels = [0] * length # Fill labels with 0.
    for ind, cla in zip(Index, Class):
        labels[ind] = cla #At the position of peak, say what type of peak it is. 
    return labels


#For peak detection only.
#Will only generate labels of 1 (peak) or 0 (not peak).
def generate_labels_detect(Index, length):
    labels = [0] * length # Fill labels with 0.
    for ind in Index:
        labels[ind] = 1 #Where there is a peak, make label 1.
    return labels


#For peak classification only.
#Will shift the peak labels from 1-5 to 0-4 as XGB wants class labels to begin at 0.
def correct_labels(targets):
    labels = []
    for item in targets:
        label = item - 1
        labels.append(label)
    return labels


#Train the XGB model
def train_classify_XGB(X_train, y_train):
    #Create XGBClassifier object
    model = XGBClassifier(use_label_encoder=False,
                          eval_metric='mlogloss',
                          max_depth=3,
                          min_child_weight=0.45,
                          subsample=0.5,
                          colsample_bytree=0.7,
                          objective= 'multi:softmax',
                          num_class = 5) 
    model.fit(X_train, y_train) 
    return model


#Test the XGB model
def test_classify_XGB(model, X_test):
    y_pred = model.predict(X_test) #Predict the response for test dataset
    predictions = [round(value) for value in y_pred] # Round all the predictions
    return predictions


# For peak detection only.
def train_detect_peak_XGB(dataset, positions):
    print('\nPeak Detection:')
    class_num = 2 # Define the number of classes
    xg_labels = generate_labels_detect(positions, len(dataset)) # Generate the labels.
    xg_points = process_positions_detect(dataset) # Generate the samples

    y, X = zip(*sorted(zip(xg_labels, xg_points))) # Sort the data by the labels in ascending order.

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(list(X), list(y), test_size=test_size, random_state=seed) # Split the data into training and test data.
    model = train_classify_XGB(X_train, y_train) # Train the model
    predictions = test_classify_XGB(model, X_test) # Test the model and produce the models predictions
    accuracy = accuracy_score(y_test, predictions) 
    print("Accuracy: %.2f%%" % (accuracy * 100.0)) # Display how many the model predicted correctly.
    fcal.calculate_multi_F1(y_test, predictions, class_num) # Display F1 score and precision scores.
    return model


#For peak classification only.
def train_classify_peak_XGB(dataset, positions, targets):
    print('\nPeak Classification:')
    class_num = 5 # Define the number of classes
    xg_input = process_positions_classify(dataset, positions) # Generate the samples.
    xg_labels = correct_labels(targets) # Generate the labels.

    y, X = zip(*sorted(zip(xg_labels, xg_input))) # Sort the data by the labels in ascending order.

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(list(X), list(y), test_size=test_size, random_state=seed) # Split the data into training and test data.
    model = train_classify_XGB(X_train, y_train) # Train the model
    predictions = test_classify_XGB(model, X_test) # Test the model and produce the models predictions
    accuracy = accuracy_score(y_test, predictions) 
    print("Accuracy: %.2f%%" % (accuracy * 100.0)) # Display how many the model predicted correctly.
    fcal.calculate_multi_F1(y_test, predictions, class_num) # Display F1 score and precision scores.
    return model


#For peak detection and classification. 
def train_classify_peak_all_XGB(dataset, positions, targets):
    print('\nPeak Detection and Classification:')
    class_num = 6 # Define the number of classes

    xg_labels = generate_labels_classify(positions, targets, len(dataset)) # Generate the labels.
    xg_points = process_positions_detect(dataset) # Generate the samples.

    y, X = zip(*sorted(zip(xg_labels, xg_points))) # Sort the data by the labels in ascending order.

    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(list(X), list(y), test_size=test_size, random_state=seed) # Split the data into training and test data.
    model = train_classify_XGB(X_train, y_train) # Train the model
    predictions = test_classify_XGB(model, X_test) # Test the model and produce the models predictions
    accuracy = accuracy_score(y_test, predictions) 
    print("Accuracy: %.2f%%" % (accuracy * 100.0)) # Display how many the model predicted correctly.
    fcal.calculate_multi_F1(y_test, predictions, class_num) # Display F1 score and precision scores.
    return model

#Shift labels from 1-5 to 0-4
def correct_labels(targets):
    unsorted_y = []
    for item in targets:
        label = item - 1
        unsorted_y.append(label)
    return unsorted_y


#Process the data to output 20x1 samples
def process_positions(dataset, positions):
    output_range = 12 # Distance that will be taken around the central point.
    classification_input = []
    for position in positions:
        lower = position - output_range
        upper = position + output_range
        classification_input.append(list(dataset[lower:upper])) # Take the section around position
    return classification_input


# Get the data
mat = spio.loadmat('training.mat')
d = mat['d']
Index = mat['Index']
Class = mat['Class'] 

model_detect = train_detect_peak_XGB(d[0], Index[0]) # Train a model to detect peaks.
model_classify = train_classify_peak_XGB(d[0], Index[0], Class[0]) # Train a model to classify peaks.Index
model_classify_full = train_classify_peak_all_XGB(d[0], Index[0], Class[0]) #Train a model to detect and classify peaks.
