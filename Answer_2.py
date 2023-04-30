import pandas as pd
import numpy as np
import os
import tensorflow as tf
from google.colab import drive
drive.mount("/content/gdrive")
from matplotlib import pyplot as plot_mp_lib
img_folder = '/content/gdrive/My Drive/MachineLearningFractal-3/GMD/train/'
IMG_HEIGHT = 32
IMG_WIDTH = 32
def create_dataset(img_folder):
img_data_array=[]
class_name=[]
for dir1 in os.listdir(img_folder):
for file in os.listdir(os.path.join(img_folder, dir1)):
img_path= os.path.join(img_folder, dir1, file)
img= cv2.imread( img_path, cv2.COLOR_BGR2GRAY)
img=np.array(img)
img = img.astype('float32')
img /= 255
img_data_array.append(img)
class_name.append(dir1)
return img_data_array, class_name
img_data, class_name = create_dataset(img_folder)
x_train = np.array(img_data).flatten
y_train = np.array(class_name).reshape(-1)
target_dict={k: v for v, k in enumerate(np.unique(class_name))}
target_val= [target_dict[class_name[i]] for i in range(len(class_name))]
model=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(IMG_HEIGHT,IMG_WIDTH)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
