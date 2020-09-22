import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense
import matplotlib.pyplot as plt
checkpoint_path = "./checkpoint/mnist.ckpt"
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())
# ])
class NumModel(Model):
    def __init__(self):
        super(NumModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        y = self.flatten(x)
        y = self.d1(y)
        y = self.d2(y)
        return y
model = NumModel()
model.load_weights(checkpoint_path)

num = int(input("check number:"))
for i in range(num):
    image_path = input("name:")
    img = Image.open(image_path)
    img = img.resize((28,28), Image.ANTIALIAS)

    img = np.array(img.convert("L"))
    # plt.imshow(img)
    # plt.show()
    img = 255 - img
    # for i in range(28):
    #     for j in range(28):
    #         if img_arr[i][j] < 200:
    #             img_arr[i][j] = 255
    #         else:
    #             img_arr[i][j] = 0
    #
    # img_arr = img_arr / 255.0

    img = img / 255.0
    x_pred = img[tf.newaxis,...]
    result = model.predict(x_pred)
    #print(result)
    result = tf.argmax(result, axis=1)
    tf.print(result)