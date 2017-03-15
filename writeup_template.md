#**Traffic Sign Recognition** 

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
[image13]: ./examples/WrongPredicted.png
[image14]: ./examples/PredictionProbabilitys.png

[image20]: ./images/12/images%20(1).jpg "priority road"
[image20P]: ./examples/TrafficSignPrediction/P1.png 
[image21]: ./images/12/images%20(3).jpg "priority road"
[image21P]: ./examples/TrafficSignPrediction/P2.png 
[image22]: ./images/12/images%20(4).jpg "priority road"
[image22P]: ./examples/TrafficSignPrediction/P3.png 
[image23]: ./images/12/images.jpg "priority road"
[image23P]: ./examples/TrafficSignPrediction/P4.png 

[image25]: ./images/14/images%20(1).jpg "stop sign"
[image25P]: ./examples/TrafficSignPrediction/S1.png 
[image26]: ./images/14/images.jpg "stop sign"
[image26P]: ./examples/TrafficSignPrediction/S2.png 
[image27]: ./images/14/stop_1.jpg "stop sign"
[image27P]: ./examples/TrafficSignPrediction/S3.png 

[image30]: ./images/15/NoVehile.jpg 
[image30P]: ./examples/TrafficSignPrediction/NP1.png 
[image31]: ./images/15/images%20(1).jpg
[image31P]: ./examples/TrafficSignPrediction/NP2.png 
[image32]: ./images/15/images.jpg 
[image32P]: ./examples/TrafficSignPrediction/NP3.png 

[image35]: ./images/2/Speed2.jpg 
[image35P]: ./examples/TrafficSignPrediction/50_1.png 
[image36]: ./images/2/Speed_flat.jpg
[image36P]: ./examples/TrafficSignPrediction/50_2.png
[image37]: ./images/2/images.jpg
[image37P]: ./examples/TrafficSignPrediction/50_3.png

[image40]: ./images/22/bum.jpg
[image40P]: ./examples/TrafficSignPrediction/B1.png 
[image41]: ./images/22/bumb2.jpg
[image41P]: ./examples/TrafficSignPrediction/B2.png 
[image42]: ./images/22/bumby.jpg
[image42P]: ./examples/TrafficSignPrediction/B3.png 



## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. [project code](https://github.com/ChristianMarzahl/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second and third code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for the barchart is contained in the fourth code cell

![alt text][image10]

The Traffic Sign distribution shows that the number of signs for each classes have a strong variance. At the data augmentation step, this problem should be addressed to prevent a classification bias at one class. 


The following image show some example traffic signs and the code is  at the fifth cell of the notebook

![alt text][image9]


### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixt code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because the paper from [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggest that color don´t have a big positive impact on the result. 
At the same time I performed a image rezise to 32x32.

To decrease the effect of different light conditions one the traffic signs I used a [local histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html) 

![alt text][image11]

As a last step after the  Keras data argumentation, I normalized the image data because thats lead to a more stable convergence of weight and biases.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The traffic sign dataset was already splitted into train, test and validation. The code for loading the data is in the first code block. 

The sixt to ninth code cell of the IPython notebook contains the code for augmenting the data set with the Keras [ImageDataGenerator](https://keras.io/preprocessing/image/).
I decided to generate additional data because the traffic sign classes where not balanced. 
To add more data to the the data set, I used the following techniques

1. rotation_range to simulate traffic signs on rising  roads with a max angle of 20°
2. width_shift_range and height_shift_range to make the classfier more robust to where the traffic sign is located insight  the image 
3. zoom_range to simulate different traffic sign sizes from the same type and that the sign ist not entirely  visible

After the data argumentation I saved the images to file for later reusability  

Here is an example of augmented images:

![alt text][image12]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the thirteenth  cell of the ipython notebook. 

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
| RELU					|												|
| Concat					|		input conv1 = 5x5x40. input conv2 = 16*16*20  Output = 4920.										|
| Fully connected		| input 4920, output 1500   									|
| Dropout					|	25%											|
| Fully connected		| input 1500, output 500   									|
| Dropout					|	25%											|
| RELU					|												|
| Softmax				| input 500, output 43        									|



#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fourteenth to sixteenth  cell of the ipython notebook. 

To train the model, I used an AdamOptimizer with a learning rate of 0.001. The batch_size was 256 and I trained for 100 Epochs. The training dropout rate was set to 0.25. For weight initialization I used the truncated_normal function with mean of 0 and standard deviation of 0.1
On my Windows machine I experienced a CuDNN exception that was not on the aws machine. That is the explanation for the line "evice_count = {'GPU': 0}" in my code to use CPU. 

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventeenth cell of the Ipython notebook.

My final model results were:
* validation set accuracy was in range of 0.982 to 0.987  
* test set accuracy of 0.965 

First I used the LeNet architecture because it was recomendet at the udacity course. I updated the model to handle the 43 output classes, better weight and bias initialisation and gray scale images.  
I increased the size and count of the FC Layer and added dropout to prevent offerfiting. The two steps increased the test accuracy by arround 2%. 
After that I tested the architecture descripted at [Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) which lead to much better results. I added a aditional FC layer because ["No hidden layer should be less than a quarter of the input layer’s nodes."](https://deeplearning4j.org/troubleshootingneuralnets) 

Additional steps to improve the accuracy could be to use inception modules to use different patch sizes at each convultional layer.  

#### The following image shows some traffic signs with wrong predictions 
![alt text][image13]
One conclusion is that even with local histogram equalization bad light condions are one of the main problems to correctly classify traffic signs. 

### Test a Model on New Images

#### 1. Choose sixthen German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are some German traffic signs that I found on the web and the results of the predictions.
The code for making predictions on my final model is located in the eighteenth and nineteenth cell of the Ipython notebook.

| Image     | Expected Difficultys | Prediction |
|:---------------------:|:---------------------------------------------:|:---------------------:|
| ![alt text][image20] | <ul><li> Watermarks </li><li> The image is smaler than the images in the trainings data  </li></ul>|![alt text][image20P] |
| ![alt text][image21] | <ul><li> Watermarks </li></ul> |![alt text][image21P] |
| ![alt text][image22] | <ul><li> Watermarks </li><li> Unusual traffic sign viewpoint  </li></ul> |![alt text][image22P] |
| ![alt text][image23] | <ul><li> Unusual position </li></ul> |![alt text][image23P] |
| ![alt text][image25] | <ul><li> No difficultys </li></ul> |![alt text][image25P] |
| ![alt text][image26] | <ul><li> No difficultys </li></ul> |![alt text][image26P] |
| ![alt text][image27] | <ul><li> No difficultys </li></ul> |![alt text][image27P] |
| ![alt text][image30] | <ul><li> Watermarks </li><li> The image is smaler than the images in the trainings data  </li></ul> |![alt text][image30P] |
| ![alt text][image31] | <ul><li> Watermarks </li></ul> |![alt text][image31P] |
| ![alt text][image32] | <ul><li> Watermarks </li></ul> |![alt text][image32P] |
| ![alt text][image35] | <ul><li> No difficultys </li></ul> |![alt text][image35P] |
| ![alt text][image36] | <ul><li> traffic sign painted on the road </li></ul> |![alt text][image36P] |
| ![alt text][image37] | <ul><li> No difficultys </li></ul> |![alt text][image37P] |
| ![alt text][image40] | <ul><li> No difficultys </li></ul> |![alt text][image40P] |
| ![alt text][image41] | <ul><li> No difficultys  </li></ul> |![alt text][image41P] |
| ![alt text][image42] | <ul><li> Watermarks </li></ul> |![alt text][image42P] |

The model was able to correctly guess 7 of the 16 traffic signs, which gives an accuracy of 37.5%. This result shows to thinks. First if the images look like the test images, traffic sign in the center of the image and a centered viewpoint are correctly classified. Second if that is not the case the classification results are not at a acceptable level. Different viewpoints is hardly to simulate with different viewpoints so only more images or no images with this viewpoints are the solution. 

