import tensorflow as tf
from sklearn import datasets
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class IrisModel(Model):
    def __init__(self):
        super(IrisModel, self).__init__()
        self.d1 = Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l1())

    def call(self, x):
        y = self.d1(x)
        return y

data = datasets.load_iris().data
target = datasets.load_iris().target

x_train = data[:-30]
y_train = target[:-30]

x_test = data[-30:]
y_test = data[-30:]

np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(target)
tf.random.set_seed(185)
model = IrisModel()
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["sparse_categorical_accuracy"])
model.fit(data, target, batch_size=32, epochs=200, validation_split=0.2, validation_freq=1)
model.summary()
