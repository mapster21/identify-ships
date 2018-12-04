# identify-ships
# Identifying Ships in Satellite Imagery
This repository contains scripts (ships.py & ships2.py) that enable the automatic detection of container ships in Planet imagery using machine learning techniques. Included are files which define a machine learning model, train it using the ShipsNet dataset, and apply it across an entire image scene to highlight ship detections.

# Input Data
JSON formatted file containing data, labels, scene id's, and location metadata:

kaggle datasets download -d rhammell/ships-in-satellite-imagery

All 4000 image clips:

https://www.kaggle.com/rhammell/ships-in-satellite-imagery#shipsnet.zip

# Setup
Python 3.5+ is required for compatability. Anaconda Prompt was used to confirm Conda was installed correctly and to insatll the new verion of python in a new environment and then activate it for use with Spyder - conda code for this is in Conda_EnvSetUpCommands.txt

# Clone this repository
git clone https://github.com/mapster21/identify-ships.git

# Go into the repository
cd identify-ships

# Install required modules
pip install -r requirements.txt

# Methodology
ShipsNet is a labeled training dataset consiting of image chips extracted from Planet satellite imagery. It contains hundreds of 80x80 pixel RGB image chips labeled with either a "ship" or "no-ship" classification. Example images are contained in the images directory.

# Ships.py
The Keras Sequential model in ships.py is a linear stack of layers. An input_shape arguement is passed to the first layer. The learning process is done via the compile method using Stochastic gradient descent (SGD) optimizer, with categorical_crossentropy string identifier and as this is a classification problem the 'accuracy' metric. The model is trained on a Numpy array of input data and labels using the fit function iterating on the data in batches of 32 samples.
	
Results graphics are contained in the results/ships directory.

# Ships2.py
Convolutional neural networks are deep artificial neural networks that are used primarily to classify images (e.g. name what they see), cluster them by similarity (photo search), and perform object recognition within scenes. The convolutional network in ships2.py 	predicts with >90% accuracy whether or not a given "image chip" contained an image of a ship.

# Model
A convolutional neural network (CNN) is defined within the ship2.py module using the sklearn library. This model supports the 80x80x3 	input dimensions of the ShipsNet image data. The Keras library was used to build a CNN.

# Training
Class MLPClassifier implements a multi-layer perceptron (MLP) algorithm that trains using Backpropagation. MLP trains on two arrays: array X of size (n_samples, n_features), which holds the training samples represented as floating point feature vectors; and array y of size (n_samples,), which holds the target values (class labels) for the training samples. The defined CNN can be trained with the JSON version of the ShipsNet dataset. The latest version of shipsnet.json is available through the ShipsNet Kaggle page, which has further information describing the dataset layout.

# Ship Classification
Several standard classifiers are compared and their K-Fold Cross-Validatiaon Accuracy returned numerically, and graphically as a boxplot. These results are then compared with the Keras CNN accuracy.

# Ships2.py Results
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

With this Keras CNN we were able to predict with >96% accuracy whether or not a given "image chip" contained an image of a ship. 				Interestingly, we had a similar results with both XGB Classifier and Keras CNN.

Results graphics are contained in the results/ships2 directory.
