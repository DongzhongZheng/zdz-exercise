import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

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

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_train[0])
# plt.show()
x_train, x_test = x_train/255, x_test/255
#print(x_train[0], y_train[0])
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
