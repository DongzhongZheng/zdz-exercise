import os

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
from PIL import Image
import matplotlib.pyplot as plt

train_path = '../class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000/'
train_txt = '../class4/MNIST_FC/mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = '../class4/MNIST_FC/mnist_image_label/mnist_x_train.npy'
y_train_savepath = '../class4/MNIST_FC/mnist_image_label/mnist_y_train.npy'

test_path = '../class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000/'
test_txt = '../class4/MNIST_FC/mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = '../class4/MNIST_FC/mnist_image_label/mnist_x_test.npy'
y_test_savepath = '../class4/MNIST_FC/mnist_image_label/mnist_y_test.npy'

def generateda(path, txt):
    f = open(txt,'r')
    contents = f.readlines()
    f.close()
    x, y_ = [],[]
    for content in contents:
        value = content.split()
        image_path = path + value[0]
        img = Image.open(image_path)
        img = np.array(img.convert("L"))
        img = img / 255.
        x.append(img)
        y_.append(value[1])
        print("Loading :"+content)
    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
    return x, y_

class NumModel(Model):
    def __init__(self):
        super(NumModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x):
        y = self.flatten(x)
        y = self.d1(y)
        y = self.d2(y)
        return y

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print("---------Loading datasets---------")
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save),28,28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print("---------generate datesets-------------")
    x_train, y_train = generateda(train_path, train_txt)
    x_test, y_test = generateda(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(185)
model = NumModel()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["sparse_categorical_accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test), validation_freq=1)
model.summary()
