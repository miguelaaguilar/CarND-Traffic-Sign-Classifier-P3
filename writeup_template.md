# **Traffic Sign Recognition Project** 

### Writeup of Dr. Miguel Aguilar
---

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/class_0.png "Class 0 Example"
[image2]: ./output_images/class_12.png "Class 12 Example"
[image3]: ./output_images/class_14.png "Class 14 Example"
[image4]: ./output_images/class_35.png "Class 35 Example"
[image5]: ./output_images/class_39.png "Class 39 Example"
[image6]: ./output_images/dist_training.png "Distribution Training Set"
[image7]: ./output_images/dist_valid.png "Distribution Validation Set"
[image8]: ./output_images/dist_test.png "Distribution Test Set"
[image9]: ./output_images/transformations.png "Data Set Augmentation"
[image10]: ./output_images/preprocessed.png "Preprocessing"
[image11]: ./output_images/architecture1.png "Architecture 1"
[image12]: ./output_images/training1.png "Training 1"
[image13]: ./output_images/architecture2.png "Architecture 2"
[image14]: ./output_images/training2.png "Training 2"
[image15]: ./output_images/new_images.png "New Images"
[image16]: ./output_images/new_images_prediction.png "New Images Prediction"
[image17]: ./output_images/softmax.png "Softmax"

The rubric with the specifications for this project can be found [here](https://review.udacity.com/#!/rubrics/481/view)

The code of the project can be found in a Jupyter Notebook [here](Traffic_Sign_Classifier.ipynb)

---

### Data Set Summary & Exploration

The following are the calculated characteristics in Python of the initial dataset provided:

```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

Here is an exploratory visualization of examples of some classes:

Class 0: Speed limit (20km/h)
![alt text][image1]

Class 12: Priority road
![alt text][image2]

Class 14: Stop
![alt text][image3]

Class 35: Ahead only
![alt text][image4]

Class 39: Keep left
![alt text][image5]

To understand how well represented is each class in the training, validation and test sets the corresponding distribution graphs are presented in the following:

![alt text][image6]

![alt text][image7]

![alt text][image8]

### Data Set Augmentation

The initial training data set was augmented with the aim to improve the performance of the implemented models in this project.

The data set was augmented by means of the following techniques:

* Brightness increase/decrease
* Rotation
* Affine transformation
* Translation

The following is an example of each of the augmentation techniques applied to a given test image:

![alt text][image9]

### Data Set Preprocessing

To make the data sets suitable for the models, the following preprocessing techniques were applied:

* Grayscale transformation:

The implemented models were designed such that they process grayscale images. Therefore, this transformation is required.

* Contrast improvement:

The contrast improvement was applied to enhance the details of the traffic signs to improve the performance of the models.

* Normalization:

The image data should be normalized so that the data has mean zero. The following are the means of the training, validation and test data sets before and after normalization: 

```
Mean values before normalization:
X_train = 99.7577245454834
X_valid = 100.90490628543084
X_test = 100.97316633573337

Mean values after normalization:
X_train = -0.20362353559102458
X_valid = -0.19833873675568914
X_test = -0.20924525446180167
```
The following are a set of examples after applying the preprocessing techniques:

![alt text][image10]

### Design and Test a Model Architecture

In this project two different model architectures were implemented and evaluated. The first is the traditional LeNet-5 from Y. LeCunn and the second is a modified architecture designed for traffic sign classification. Both are described in the following subsections.

#### 1. Model Architecture: LetNet-5 - Traditional Architecture

The first architecture implemented in this project is the LetNet-5 proposed by Y. LeCunn et al. in the paper ["Gradient-based learning applied to document recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) in 1998. This is the traditional ConvNet architecture used as a reference during the lessons 14 y 15 of the Self-Driving Engineer Nanodegree of Udacity. The architecture is presented in the following figure.

![alt text][image11]
Source: ["Gradient-based learning applied to document recognition"](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

The initial architecture did not included the dropout, but after adding this technique the performance of the model increased significantly. The final model is based on the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 				|
| Fully connected		     | input 400, output 120  									|
| RELU					             |												|
| Dropout		             |												|
| Fully connected		     | input 120, output 84  									|
| RELU					             |												|
| Dropout		             |												|
| Softmax				           | input 84, output 43  									|

To evaluate this model the following hyperparameters were used:

```
epochs = 80
batch size = 128
mu = 0
sigma = 0.1
learning rate = 0.0009
dropout = 0.5
```

The following is the accuracy of the validation set along the epochs

![alt text][image12]

The obtained accuracy of test set was ***95.2%***

#### 2. Model Architecture: Adapted Architecture for Traffic Sign Recognition

The second architecture implemented in this project is based on the one proposed by Pierre Sermanet and Yann LeCun in the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) in 2011. This architecture is based on the tradicional LetNet-5 implemented and evaluated in the previous section. The key difference of the adapted architecture is that the output of the first stage is branched out and fed to the classifier, in addition to the output of the second stage. The architecture is presented in the following figure.

![alt text][image13]
Source: [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

The final model is based on the following layers:

| Layer         		      |     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 grayscale image   							| 
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					             |												|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16 				|
| Convolution 5x5     	 | 1x1 stride, valid padding, outputs 1x1x400 	|
| Flatten             	 | outputs 2000 	|
| Dropout		             |												|
| Fully connected		     | input 2000, output 84  									|
| RELU					             |												|
| Dropout		             |												|
| Softmax				           | input 84, output 43  									|

To evaluate this model the following hyperparameters were used:

```
epochs = 80
batch size = 128
mu = 0
sigma = 0.1
learning rate = 0.0009
dropout = 0.5
```

The following is the accuracy of the validation set along the epochs

![alt text][image14]

The obtained accuracy of test set was ***96.4%***, which is a better performance than the traditional model architecture.

### Test a Model on New Images

Since I am located in Germany, I was able to go out to the street and take a few pictures of traffic signs as shown in the following figure.

![alt text][image15]

Then, this set of new images was classified using the adapted architecture model, since this one had the best performance. As shown in the following figure, the model correctly classified all images, i.e., achieved an accuracy of ***100%***, which is higher than the one achived with the test set.

![alt text][image16]

The softmax probabilities are shown in the following image. As can be observed, the model performance solid classification decisions in all the 6 images, since the probabilities for the right choices were around ***100%***.

![alt text][image17]


### Discussion

#### 1. Summary

In this project, it was implemented an traffic sign classifier using CNNs. Two different architectures were implemented. The first one is the traditional LeNet-5 that achieved a ***95.2** accuracy on the test set, and the second is an adapted architecture that achieved a ***96.4*** accuracy on the test set.

During the implementation and evaluation of both models it was observed that the main aspects that impacts the performance are the size of the training set, the use of dropout and the architecture.

#### 2. Possible Improvements

One obvious approach to improve even further the performane of the models is to increase the training dataset. However, since this project was carried out within a workspace of Udacity with limited GPU quota, it was not possible to train the model with extremly large dataset.

#### References

* Part of the code of this project was taken from lessons of the Self Driving Car Engineer Nanodegree of Udacity
* Some of image transformation functions for the dataset augmentation were adapted from [here](https://github.com/sharathsrini/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier_Final.ipynb). 