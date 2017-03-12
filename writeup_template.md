#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image9]: ./examples/original_traffic_signs.png "Traffic Sign Overview"
[image10]: ./examples/traffic_sign_count_overview.png "Traffic Sign Overview Count"
[image11]: ./examples/gray_adaptive_hist.png "gray_adaptive_hist"
[image12]: ./examples/DataArgumentationGray.png "DataArgumentationGray"

[image20]: ./images/12/images (1).jpg "priority road"
[image21]: ./images/12/images (3).jpg "priority road"
[image22]: ./images/12/images (4).jpg "priority road"
[image23]: ./images/12/images.jpg "priority road"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. [project code](https://github.com/ChristianMarzahl/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second and third code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for the barchart is contained in the fourth code cell

![alt text][image10]

The Traffic Sign distribution shows that the number of signs for each classes have a strong variance. At the data augmentation step, this problem should be addressed to prevent a classification bias at one class. 


The following image show some example traffic signs and the code is  at the fifth cell of the notebook

![alt text][image9]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixt code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the paper from [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggest that color don´t have a big positive impact on the result. 
At the same time I performed a image rezise to 32x32.

To decrease the effect of different light conditions one the traffic signs I used a [local histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) 

![alt text][image11]

As a last step after the  Keras data argumentation, I normalized the image data because thats lead to a more stable convergence of weight and biases.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The traffic sign dataset was already splitted into train, test and validation. The code for loading the data is in the first code block. 

The * * * code cell of the IPython notebook contains the code for augmenting the data set with the Keras [ImageDataGenerator](https://keras.io/preprocessing/image/).
I decided to generate additional data because the traffic sign classes where not balanced. 
To add more data to the the data set, I used the following techniques

1. rotation_range to simulate traffic signs on rising  roads with a max angle of 20°
2. width_shift_range and height_shift_range to make the classfier more robust to where the traffic sign is located insight  the image 
3. zoom_range to simulate different traffic sign sizes from the same type and that the sign ist not entirely  visible

After the data argumentation I saved the images to file for later reusability  

Here is an example of augmented images:

![alt text][image12]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the * * *  cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x20 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x40 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x40 				|
| Fully connected		| input 1000, output 500   									|
| RELU					|												|
| Fully connected		| input 500, output 300   									|
| Dropout					|	25%											|
| RELU					|												|
| Fully connected		| input 300, output 100   									|
| Dropout					|	25%											|
| RELU					|												|
| Softmax				| input 100, output 43        									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the * * *  cell of the ipython notebook. 

To train the model, I used an AdamOptimizer with a learning rate of 0.001. The batch_size was 128 and I trained for 100 Epochs. The training dropout rate was set to 0.25. For weight initialization I used the truncated_normal function with mean of 0 and standard deviation of 0.1
On my Windows machine I experienced a CuDNN exception that was not on the aws machine. That is the explanation for the line "evice_count = {'GPU': 0}" in my code to use CPU. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the * * * * cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

First I used the LeNet architecture because it was recomendet at the udacity course. I updated the model to handle the 43 output classes, better weight and bias initialisation and gray scale images.  

I increased the size of the FC Layer and added dropout to prevent offerfiting. The two steps increased the test accuracy by arround 2%. 

Additional steps to improve the accuracy could be to add more Conv layers or use inception modules to use different patch sizes.  


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web:

![alt text][image20] ![alt text][image21] ![alt text][image22] 
![alt text][image23] ![alt text][image24]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
