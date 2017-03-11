# Load pickled data
import pickle
import numpy as np
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'files/train.p'
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

print (X_train.shape)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.

#Plotting the bar graph of the frequency of classes 
plt.barh(ids, counts, align='center', alpha=0.5)
plt.yticks(ids, id_labels)
plt.title('Trafic sign class distribution')
plt.xlabel('Count')
plt.show()

sample_per_class = [np.where(y_train==x)[0] for x in range(n_classes)]

fig, axes = plt.subplots(8, 6, figsize=image_shape,
                         subplot_kw={'xticks': [], 'yticks': []})

for ax, index_list in zip(axes.flat,sample_per_class):
    index = np.random.choice(index_list)
    ax.imshow(X_train[index].squeeze())
    ax.set_title(signnames.loc[y_train[index]][0])

plt.show()



# Step 2: Design and Test a Model Architecture
# Keras for data argumentation 

import cv2
# convert the the training images to grayscale
# according to the referenced paper that leads to better results 
X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).reshape(32,32,1) for image in X_train])

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
K.set_image_dim_ordering('tf') #tf
import cv2


# performs data argumentation 
datagen = ImageDataGenerator(
    #featurewise_center=True, #Set input mean to 0 over the dataset, feature-wise.
    #featurewise_std_normalization=True, # Divide inputs by std of the dataset, feature-wise.
    rotation_range=20, #Degree range for random rotations.
    width_shift_range=0.2, # Range for random horizontal shifts.
    height_shift_range=0.2, #  Range for random vertical shifts.
    #zca_whitening=True, # Apply ZCA whitening.
    zoom_range=0.3,
    ) #Range for random zoom

#compute quantities required for featurewise normalization
#(std, mean, and principal components if ZCA whitening is applied)
X_train = X_train.astype('float32')
datagen.fit(X_train)

# find the traffic sign which has the most examples and generate with data argumentation  
# equal counts for the other traffic signs 
target_imges_per_class = max([len(array) for array in sample_per_class])

X_train_neu = X_train
y_train_neu = y_train

for samples in sample_per_class:
    generated_images_per_class_counter = 0
    batch_size = 32
    for X_batch, Y_batch in datagen.flow(X_train[samples], y_train[samples], batch_size=batch_size):
        X_train_neu = np.vstack((X_train_neu,X_batch))
        y_train_neu = np.hstack((y_train_neu,Y_batch))

        generated_images_per_class_counter += batch_size

        if target_imges_per_class - len(samples) < generated_images_per_class_counter:
            break

# Save the generated signs to file for reuse 
train_augmented_file_path = "./files/train_augmented_gray.p" 
with open(train_augmented_file_path, 'wb') as train_augmented_file:
    output_dict = {'features': 1, 'labels': 2}
    output_dict['features'] = X_train_neu
    output_dict['labels'] = y_train_neu
    pickle.dump(output_dict, train_augmented_file)
train_augmented_file.close()

sample_per_class = [np.where(y_train_neu==x)[0] for x in range(n_classes)]

ids, index, counts = np.unique(y_train_neu, return_index = True, return_counts=True)
id_labels = [signnames.loc[x][0] + " [{}]".format(x) for x in ids]
n_classes = len(ids)

#Plotting the bar graph of the frequency of classes 
plt.barh(ids, counts, align='center', alpha=0.5)
plt.yticks(ids, id_labels)
plt.title('Trafic sign class distribution')
plt.xlabel('Count')
plt.show()



