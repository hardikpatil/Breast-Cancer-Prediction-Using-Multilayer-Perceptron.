# Breast-Cancer-Prediction-Using-Multilayer-Perceptron.
Breast Cancer Prediction Using Multilayer Perceptron.
Problem statement
Breast cancer represents one of the diseases that make a high number of deaths every year. It is
the most common type of all cancers and the main cause of women's deaths worldwide.
Classification and data mining methods are an effective way to classify data. Especially in
medical field, where those methods are widely used in diagnosis and analysis to make decisions.
We have performed analysis using machine learning algorithm : Multilayer Perceptron on the
Wisconsin Breast Cancer (original) datasets. The main objective is to assess the correctness in
classifying data with respect to efficiency and effectiveness of each algorithm in terms of
accuracy, precision, sensitivity and specificity.
Our purpose is to train the model from the given dataset so that the diagnosis/prediction is either benign
(non- cancerous) tumor or malignant tumor (cancerous).
Introduction
We have collected the dataset from:
http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29
We have create a web application comprising of three functionalities viz. Analysis visualization
using pie charts, to train the model and check the accuracy and to predict the class of tumors on
inputs by the user if it is malignant or benign.
Dataset
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
They describe characteristics of the cell nuclei present in the image.
Characteristics of dataset:
1. Multivariate dataset
2. No. of instances : 569
3. Area : Life
4. Attribute Characteristics: Real
5. No. of attributes : 32
6. Associated task : Classification
7. Missing values : No
Attribute Information:
1. ID number
2. Diagnosis (M = malignant, B = benign)
Ten real-valued features are computed for each cell nucleus:
1. radius (mean of distances from center to points on the perimeter)
2. texture (standard deviation of gray-scale values)
3. perimeter
4. area
5. smoothness (local variation in radius lengths)
6. compactness (perimeter^2 / area - 1.0) g) concavity (severity of concave portions of the
contour)
7. concave points (number of concave portions of the contour)
8. symmetry
9. fractal dimension ("coastline approximation" - 1)
The mean, standard error and "worst" or largest (mean of the three largest values) of these
features were computed for each image, resulting in 30 features. For instance, field 3 is Mean
Radius, field 13 is Radius SE, field 23 is Worst Radius.
All feature values are recorded with four significant digits.
Missing attribute values: none
Class distribution: 357 benign, 212 malignant
Algorithm
We have used neural network algorithm called Multilayer Perceptron (MLP). The reason behind
using this algorithm was that MLPs are useful in research for their ability to solve problems
stochastically, which often allows approximate solutions for extremely complex problems like
fitness approximation .MLPs are universal function approximators as showed by Cybenko's
theorem, so they can be used to create mathematical models by regression analysis. As
classification is a particular case of regression when the response variable is categorical , MLPs
make good classifier algorithms.
A multilayer perceptron (MLP) is a class of feedforward artificial neural network . An MLP
consists of at least three layers of nodes. Except for the input nodes, each node is a neuron that
uses a nonlinear activation function . MLP utilizes a supervised learning technique called
backpropagation for training. Its multiple layers and non-linear activation distinguish MLP from
linear perceptron . It can distinguish data that is not linearly separable .
Fig1. A hypothetical example of Multilayer Perceptron
Activation function
If a multilayer perceptron has a linear activation function in all neurons, that is, a linear function
that maps the weighted inputs to the output of each neuron, then linear algebra shows that any
number of layers can be reduced to a two-layer input-output model. In MLPs some neurons use a
nonlinear activation function that was developed to model the frequency of action potentials , or
firing, of biological neurons.
The two common activation functions are both sigmoids , and are described by
The first is a hyperbolic tangent that ranges from -1 to 1, while the other is the logistic function ,
which is similar in shape but ranges from 0 to 1. Here yi is the output of the ith node (neuron)
and vi is the weighted sum of the input connections. Alternative activation functions have been
proposed, including the rectifier and softplus functions. More specialized activation functions
include radial basis functions (used in radial basis networks, another class of supervised neural
network models).
Layers
The MLP consists of three or more layers (an input and an output layer with one or more hidden
layers ) of nonlinearly-activating nodes making it a deep neural network . Since MLPs are fully
connected, each node in one layer connects with a certain weight wij to every node in the
following layer.
Learning
Learning occurs in the perceptron by changing connection weights after each piece of data is
processed, based on the amount of error in the output compared to the expected result. This is an
example of supervised learning , and is carried out through backpropagation , a generalization of
the least mean squares algorithm in the linear perceptron.
This depends on the change in weights of the kth nodes, which represent the output layer. So to
change the hidden layer weights, the output layer weights change according to the derivative of
the activation function, and so this algorithm represents a backpropagation of the activation
function.
Analysis
The project is a classification based model. It uses the mean, standard error and worst case of the
ten parameters discussed above to classify a tumour as malignant (cancerous) or benign (non
cancerous).
The MultiLayer Perceptron model gives a accuracy between 92-99%. The reason for varied
accuracy for the same dataset is that the model divides the dataset into training and testing parts.
80% of the dataset is used for training and 20% for testing. So each time the model chooses
different tuples for training and testing and it is not kept constant. In a way, each time a new
training and testing dataset is generated even with the same csv file being used.
This can be observed in the GUI output.
Parameters specified for MLP Classifier:
1. Hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden
layer. Specified as 45.
2. activation : Activation function for the hidden layer. Specified as ‘relu’, the rectified
linear unit function, returns f(x) = max(0, x)
3. solver : The solver for weight optimization. Specified as ‘lbfgs’ an optimizer in the
family of quasi-Newton methods.
4. alpha :L2 penalty (regularization term) parameter. Specified as e-10.
5. learning_rate : specified as ‘constant’, a constant learning rate given by
‘learning_rate_init’.
6. random_state :random_state is the seed used by the random number generator. Specified
as 1.
Software Specification
1. Python 3.6
2. Django 2.0.3
Tools Used
1. Scikit- sklearn MLP Classifier
GUI Screenshots
Fig 2. Analysis Module
Fig 3. Prediction Module
Fig 4. Training and Testing Module
Conclusion
Thus we have trained our model for detecting whether the tumor is malignant (cancerous) or benign
(non-cancerous) using Multi Layer Perceptron Classifier, i.e. a neural network model. The application is
developed in python using the django framework. We managed to achieve an accuracy in the range
92-99%.
