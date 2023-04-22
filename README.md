# peak_classifier_and_detector

Inside training.mat, there is d, Index and Class. d is a dataset. Index provides the positions of the peaks within the dataset and Class defines which class each peak is in. 
This repository provides 2 solutions to detect and classify peaks within d. The first solution uses the xgboost library. The second solution uses a simple decision tree. 

F1 calculator is used to calculate the accuracy of the solutions.


Packages to be downloaded:

xgboost


Xgboost is used to detect and classify the peaks. Xgboost is a library that uses machine learning algorithms to provide paralell tree boosting.
