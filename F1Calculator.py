import numpy as np


# Calculate a binary matrix. Use only for peak detector.
def calculate_confusion_matrix(length_data, referenced_positives, predicted_positives):
    true_pos = 0 
    false_neg = 0
    #For each actual positive:
    for ref in referenced_positives:
        #Comparing the actual positive to all the predicted positives.
        for pre in predicted_positives:
            if ref == pre:
                #if actual and predicted are the same, it is a true positive.
                true_pos +=1 
            elif abs(ref-pre) < 50:
                #We will be marked if it is within 50 positions, the peak positons is correct.
                diff = abs(ref-pre)
                ref_found = True
                # Check to see if the predicted point is closest to the actual point.
                for ref_pos in referenced_positives:
                    if diff > abs(ref_pos-pre):
                        ref_found = False
                        break
                if ref_found == True:
                    true_pos +=1
                else:
                    false_neg += 1
            else:
                #if there is nothing like the actual point, it is a false negative
                false_neg += 1
    #Calculate the other values.
    false_pos = len(predicted_positives) - true_pos
    false_neg = len(referenced_positives) - true_pos
    true_neg = length_data - true_pos - false_pos - false_neg
    confusion_matrix = [true_pos, true_neg, false_pos, false_neg]
    print(confusion_matrix)
    return confusion_matrix


#Calculate precision values.
def calculate_precision(confusion_matrix):
    precision = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[2])
    return precision


#Calculate recall values.
def calculate_recall(confusion_matrix):
    recall = confusion_matrix[0]/(confusion_matrix[0]+confusion_matrix[3])
    return recall


#Calculate F1_score
def calculate_F1score(dataset, referenced_positives, predicted_positives):
    confusion_matrix = calculate_confusion_matrix(dataset, referenced_positives, predicted_positives)
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    F1_score = (2*precision*recall)/(precision+recall)
    print("Precision:", precision)
    print("F1_score:", F1_score)
    return F1_score


#Calculate a multi-class confusion matrix
def calculate_multi_confusion_matrix(ref_classes, pre_classes, length):
    confusion_matrix = np.zeros((length,length))
    for ref,pre in zip(ref_classes, pre_classes):
        #Predicted is the rows, refernce is the columns.
        confusion_matrix[pre, ref] += 1
    return confusion_matrix


#Calculate precision and recall for each class.
def calculate_multi_pre_recall(multi_confusion_matrix):
    class_dict = {} # Store values in a dictionary
    #Find values for confusion matrix.
    for i in range(0, len(multi_confusion_matrix[0])):
        row = multi_confusion_matrix[i] 
        true_pos = row[i]
        row_sum = np.sum(row)
        false_pos = row_sum - true_pos # The sum of all the predicted of the class minus the true positives are the false positives.
        col_sum = 0
        for row in multi_confusion_matrix:
            col_sum += row[i]
        false_neg = col_sum - true_pos # The sum of all the actual of the class minus the true positives are the false negatives.
       
        # Create a binary confusion matrix from the extracted values. 
        confusion_matrix = [true_pos, None, false_pos,  false_neg]
        precision = calculate_precision(confusion_matrix)
        recall = calculate_recall(confusion_matrix)
        class_dict[i] = [precision, recall]
    return class_dict


# Calculate F1 scores for multi-class classification
def calculate_multi_F1(ref_classes, pre_classes, length):
    multi_confusion_matrix = calculate_multi_confusion_matrix(ref_classes, pre_classes, length) # Calculate confusion matrix
    print('Confusion Matrix:')
    print(multi_confusion_matrix)
    precision_recall_dict = calculate_multi_pre_recall(multi_confusion_matrix) # Calculate precision and recall values.
    pre_sum = 0
    sum = 0
    #For each class:
    for i in range(0,len(multi_confusion_matrix[0])):
        weight = ref_classes.count(i) #Total samples in that class is the weight.
        precision = precision_recall_dict[i][0]
        recall = precision_recall_dict[i][1]
        F1_score = (2*precision*recall)/(precision+recall) # Calculate the F1 score.
        print(i,':', precision, F1_score)
        print('Weight:', weight)
        #Sum all the weighted precisions and weighted F1 scores.
        pre_sum += weight * precision 
        sum += weight * F1_score
    #Calculate the weighted precision and weighted F1 score.
    total_pre = pre_sum/ len(ref_classes)
    total_F1 = sum/ len(ref_classes)
    print('Weighted Precision:', total_pre)
    print('Weighted F1:', total_F1)
