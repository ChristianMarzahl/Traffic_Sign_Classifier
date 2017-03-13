# Load pickled data
import pickle
import numpy as np
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'files/train_augmented_gray.p'
validation_file = 'files/valid.p'
testing_file = 'files/test.p'

signnames_file = 'files/signnames.csv'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

signnames = pd.read_csv(signnames_file,index_col='ClassId') # ,index_col='ClassId'
    
# 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# 'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
# 'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
# 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = (X_train.shape[1],X_train.shape[2])
image_channels = X_train.shape[3]

# TODO: How many unique classes/labels there are in the dataset.
ids, index, counts = np.unique(y_train, return_index = True, return_counts=True)
id_labels = [signnames.loc[x][0] + " [{}]".format(x) for x in ids]
n_classes = len(ids)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Channels =", image_channels)
print("Number of classes =", n_classes)


# Step 2: Design and Test a Model Architecture


import cv2
# convert the the training images to grayscale
# according to the referenced paper that leads to better results 
X_valid = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_valid])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_test])

X_valid.astype('float32')
X_valid.astype('float32')
X_test.astype('float32')

X_train = np.array(X_train / 255.0 - 0.5)
X_valid = np.array(X_valid / 255.0 - 0.5 )
X_test  = np.array(X_test / 255.0 - 0.5 )


import tensorflow as tf
from tensorflow.contrib.layers import flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('tf') #tf


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
    #Pooling. Input = 28x28x20. Output = 14x14x20.
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

    #Flatten. Input1 = 5x5x40. Input2 = 16*16*20  Output = 4920.
    fc0   = tf.concat(1, [flatten(conv2),flatten(conv1)])

    
    #Layer3  4920 --> 1500
    fc1_W = tf.Variable(tf.truncated_normal(shape = (4920,1500), mean = mu , stddev = sigma), name = 'fc1_W')
    fc1_b = tf.Variable(tf.truncated_normal(shape=[1500], mean = mu, stddev = sigma), name = 'fc1_b')
    fc1 = tf.add(tf.matmul(fc0, fc1_W) , fc1_b)  
     #Dropout 
    fc1 = tf.nn.dropout(fc1,keep_prob)    
    fc1 = tf.nn.relu(fc1) 
  
    #Layer5 1500 --> 500
    fc2_W = tf.Variable(tf.truncated_normal(shape = (1500,500), mean = mu, stddev = sigma), name = 'fc2_W')
    fc2_b = tf.Variable(tf.truncated_normal(shape=[500], mean = mu, stddev = sigma), name = 'fc2_b')
    fc2 = tf.add(tf.matmul(fc1,fc2_W) , fc2_b)
    #Dropout 
    fc2 = tf.nn.dropout(fc2,keep_prob)
    fc2 = tf.nn.relu(fc2)


    #Layer6 100 --> 43
    fc3_W = tf.Variable(tf.truncated_normal(shape = (500,n_classes), mean = mu, stddev = sigma), name = 'fc3_W')
    fc3_b = tf.Variable(tf.truncated_normal(shape=[n_classes], mean = mu, stddev = sigma), name = 'fc3_b')
    logits = tf.add(tf.matmul(fc2,fc3_W) , fc3_b)

    return logits


# Features and Labels
# traffic sign consists of 32x32x3, images
x = tf.placeholder(tf.float32, (None, image_shape[0],image_shape[1], image_channels))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y,n_classes)
image_channels = tf.constant(image_channels)
n_classes = tf.constant(n_classes)

# Training Pipeline
rate = 0.001
logits = LeNet(x,image_channels,n_classes,keep_prob)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Model Evaluation

BATCH_SIZE = 256
EPOCHS = 100

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

# Train the Model
saver = tf.train.Saver()
save_file = './classifier/lenet.ckpt'

from sklearn.utils import shuffle
config = tf.ConfigProto(
        #ädevice_count = {'GPU': 0}
    )
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print ("Start Training")
    for i in range(EPOCHS):
        
        X_train, y_train = shuffle(X_train,y_train)

        for offset in range(0, num_examples,BATCH_SIZE):
            #print("BATCH {} von {} ".format(offset,num_examples))
            batch_x, batch_y = X_train[offset:offset+BATCH_SIZE], y_train[offset:offset+BATCH_SIZE]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob : 0.25})

        val_loss, val_acc = evaluate(X_valid,y_valid)

        print("EPOCH {} ...".format(i+1))
        print("Validation loss = {:.3f}".format(val_loss))
        print("Validation accuracy = {:.3f}".format(val_acc))

        if(last_val_acc < val_acc):
             last_val_acc = val_acc
             saver.save(sess,save_file)
             print('Model saved')
        print()


with tf.Session(config=config) as sess:
    saver.restore(sess, save_file)

    test_loss, test_acc = evaluate(X_test, y_test)
    print("Test loss = {:.3f}".format(test_loss))
    print("Test accuracy = {:.3f}".format(test_acc))
