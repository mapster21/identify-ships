# identify-ships
# Identifying Ships in Satellite Imagery
This repository contains scripts that enable the automatic detection of container ships in Planet imagery using machine learning techniques. Included are files which define a machine learning model, train it using the ShipsNet dataset, and apply it across an entire image scene to highlight ship detections.

# Methodology
ShipsNet is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains hundreds of 80x80 pixel RGB image chips labeled with either a "ship" or "no-ship" classification. 

Example images are contained in the images directory.

Convolutional neural networks are deep artificial neural networks that are used primarily to classify images (e.g. name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. 

This convolutional network predicts with >90% accuracy whether or not a given "image chip" contained an image of a ship.

# Setup
Python 3.5+ is required for compatability

# Clone this repository
git clone https://github.com/mapster21/identify-ships.git

# Go into the repository
cd identify-ships

# Install required modules
pip install -r requirements.txt

# Model
A convolutional neural network (CNN) is defined within the ship2.py module using the sklearn library. This model supports the 80x80x3 input dimensions of the ShipsNet image data. The Keras library was used to build a CNN.

# Training
Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation. MLP trains on two arrays: array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors; and array y of size (n_samples,), which holds the target values (class labels) for the training samples. The defined CNN can be trained with the JSON version of the ShipsNet dataset. The latest version of shipsnet.json is available through the ShipsNet Kaggle page, which has further information describing the dataset layout.

# Ship Classification
Several standard classifiers are compared and their K-Fold Cross-Validatiaon Accuracy returned numerically, and graphically as a boxplot. These results are then compared with the Keras CNN accuracy.

# Results
LR, Logistic Regression:          0.891562 (0.008155)

RF, Random Forest Classifier:     0.936875 (0.016044)

SVM, Support Vector Machine SVC:  0.748437 (0.022240)

LSVM, Linear SVC:                 0.886562 (0.010648)

GNB, Gaussian NB:                 0.631563 (0.028417)

DTC, Decision Tree Classifier:    0.900625 (0.019304)

XGB, XGB Classifier:              0.959063 (0.010866)


Keras CNN #1C - accuracy: 0.9675 


              precision    recall  f1-score   support

     No Ship       0.98      0.98      0.98       605
        Ship       0.94      0.92      0.93       195

   micro avg       0.97      0.97      0.97       800
   macro avg       0.96      0.95      0.96       800
weighted avg       0.97      0.97      0.97       800

With this Keras CNN we were able to predict with >96% accuracy whether or not a given "image chip" contained an image of a ship. Interestingly, we had a similar results with both XGB Classifier and Keras CNN.

Results graphics are contained in the results directory.
