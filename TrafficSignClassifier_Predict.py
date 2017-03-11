# Load pickled data
import pickle
import numpy as np
import pandas as pd
import cv2

signnames_file = 'files/signnames.csv'
testing_file = 'files/test.p'

with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

signnames = pd.read_csv(signnames_file,index_col='ClassId') 


X_test, y_test = test['features'], test['labels']
ids, index, counts = np.unique(y_test, return_index = True, return_counts=True)
id_labels = [signnames.loc[x][0] + " [{}]".format(x) for x in ids]
n_classes = len(ids)


X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_test])

X_test.astype('float32')
X_test  = np.array(X_test / 255.0 - 0.5 )

image_shape = (X_test.shape[1],X_test.shape[2])
image_channels = X_test.shape[3]

import tensorflow as tf
from tensorflow.contrib.layers import flatten

#Model Architecture
#Model Architecture
def LeNet(x, image_channels, n_classes,keep_prob):

    mu = 0
    sigma = 0.1

    #Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x20.
    # CONV => RELU => POOL
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_channels, 20), mean = mu, stddev = sigma), name = 'conv1_W')
    conv1_b = tf.Variable(tf.truncated_normal(shape=[20], mean = mu, stddev = sigma), name = 'conv1_b')
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    #Activation.
    conv1 = tf.nn.relu(conv1)
    #Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Layer 2:
    # CONV => RELU => POOL 14x14x20 ---> 10*10*40
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 20, 40), mean = mu, stddev = sigma), name = 'conv2_W')
    conv2_b = tf.Variable(tf.truncated_normal(shape=[40], mean = mu, stddev = sigma), name = 'conv2_b')
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    #Activation.
    conv2 = tf.nn.relu(conv2)
    #Pooling. Input = 10x10x40. Output = 5x5x40.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    #Flatten. Input = 5x5x40. Output = 1000.
    fc0   = flatten(conv2)
    
    #Layer3  1000 --> 500
    fc1_W = tf.Variable(tf.truncated_normal(shape = (1000,500), mean = mu , stddev = sigma), name = 'fc1_W')
    fc1_b = tf.Variable(tf.truncated_normal(shape=[500], mean = mu, stddev = sigma), name = 'fc1_b')
    fc1 = tf.add(tf.matmul(fc0, fc1_W) , fc1_b)   
    fc1 = tf.nn.relu(fc1) 
  
    #Layer4 500 --> 300
    fc2_W = tf.Variable(tf.truncated_normal(shape = (500,300), mean = mu , stddev = sigma), name = 'fc2_W')
    fc2_b = tf.Variable(tf.truncated_normal(shape=[300], mean = mu, stddev = sigma), name = 'fc2_b')
    fc2 = tf.add(tf.matmul(fc1,fc2_W) , fc2_b)    
    #Dropout 
    fc2 = tf.nn.dropout(fc2,keep_prob)
    fc2 = tf.nn.relu(fc2)
    
    #Layer5 300 --> 100
    fc3_W = tf.Variable(tf.truncated_normal(shape = (300,100), mean = mu, stddev = sigma), name = 'fc3_W')
    fc3_b = tf.Variable(tf.truncated_normal(shape=[100], mean = mu, stddev = sigma), name = 'fc3_b')
    fc3 = tf.add(tf.matmul(fc2,fc3_W) , fc3_b)
    #Dropout 
    fc3 = tf.nn.dropout(fc3,keep_prob)
    fc3 = tf.nn.relu(fc3)


    #Layer6 100 --> 43
    fc4_W = tf.Variable(tf.truncated_normal(shape = (100,n_classes), mean = mu, stddev = sigma), name = 'fc4_W')
    fc4_b = tf.Variable(tf.truncated_normal(shape=[n_classes], mean = mu, stddev = sigma), name = 'fc4_b')
    logits = tf.add(tf.matmul(fc3,fc4_W) , fc4_b)

    return logits, conv1_W


x = tf.placeholder(tf.float32, (None, image_shape[0],image_shape[1], image_channels))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y,n_classes)
keep_prob = tf.placeholder(tf.float32)
image_channels = tf.constant(image_channels)
n_classes = tf.constant(n_classes)

rate = 0.001
logits, conv1_W = LeNet(x,image_channels,n_classes,keep_prob)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 512
EPOCHS = 10

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_acc = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples,BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, acc  = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y,keep_prob : 1.0})
        total_acc += (acc * batch_x.shape[0])
        total_loss += (loss * batch_x.shape[0])
    return total_loss/num_examples, total_acc/num_examples



saver = tf.train.Saver()
save_file = './classifier/lenet.ckpt'

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
# Launch the graph
#with tf.Session(config=config) as sess:
#    saver.restore(sess, save_file)

#    test_loss, test_acc = evaluate(X_test, y_test)

#    print("Test loss = {:.3f}".format(test_loss))
#    print("Test accuracy = {:.3f}".format(test_acc))

from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

imageDirectory = "./images"
traffic_sign_keys = [14,15,2,22,12]
# [1,[img.png, img2.png]]
files_to_test = [[f,listdir(join(imageDirectory, str(f)))] for f in traffic_sign_keys]




### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation,activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it maybe having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input}) #
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.show()




# Output Top 5 Softmax Probabilities For Each Image Found on the Web
with tf.Session(config=config) as sess:
    saver.restore(sess, save_file)
    
    test_traffic_sign_Count = 0
    correct_count = 0

    for key, image_path_list in files_to_test:
        for image_path in image_path_list:

            test_traffic_sign_Count += 1

            image_path_with_Dict = join(imageDirectory, str(key), image_path)

            image = cv2.imread(image_path_with_Dict)
            image = cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA)
            
            # dispaly image
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1)
            X_test  = np.array([np.array(image_gray / 255.0 - 0.5 )])

            probability, classes = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k = 5), feed_dict={x: X_test, keep_prob: 1.0})

            probability = probability[0]
            classes = classes[0]

            if(classes[0] == key):
                correct_count += 1
                
            five_list = [0,1,2,3,4]
            plt.xlabel(signnames.loc[key][0], fontsize=15)
            plt.subplot(2,1,1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    
            plt.subplot(2,1,2)
            plt.barh(five_list, probability, height=0.5, align='center')
            plt.yticks(five_list,[signnames.loc[classes[i]][0] for i in five_list] )
            plt.show()

    print("Accuracy: {}".format(correct_count/test_traffic_sign_Count*100))