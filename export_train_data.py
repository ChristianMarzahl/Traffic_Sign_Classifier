import pickle
import numpy as np
import pandas as pd

training_file = 'files/train.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X_train, y_train = train['features'], train['labels']

folder_basic_path = "./images"

import os
import cv2
import uuid
for image, label in zip(X_train,y_train):

    folder_path = os.path.join(folder_basic_path,str(label))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = str(uuid.uuid4()) + ".jpg"
    file_path = os.path.join(folder_path,file_name)

    cv2.imwrite(file_path,image)
