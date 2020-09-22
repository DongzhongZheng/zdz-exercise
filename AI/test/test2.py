import tensorflow as tf
from sklearn import datasets
import numpy as np

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
tf.random.set_seed(116)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(3, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())])
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["sparse_categorical_accuracy"])
model.fit(data, target, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)
model.summary()
