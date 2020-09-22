import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import os

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
np.set_printoptions(threshold=np.inf)
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
checkpoint_savepath = "./checkpoint/mnist.ckpt"
model = NumModel()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["sparse_categorical_accuracy"])
if os.path.exists(checkpoint_savepath+".index"):
    print("--------------load widget------------")
    model.load_weights(checkpoint_savepath)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_savepath,
                                                 save_weights_only=True, save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test,y_test), validation_freq=1, callbacks=[cp_callback])
model.summary()

f = open("./weights.txt","w")
for v in model.trainable_variables:
    f.write(str(v.name)+"\n")
    f.write(str(v.shape)+"\n")
    f.write(str(v.numpy())+"\n")
f.close()

acc = history.history["sparse_categorical_accuracy"]
val_acc = history.history["val_sparse_categorical_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1,2,1)
plt.plot(acc, label="train_accuracy")
plt.plot(val_acc, label="test_accuracy")
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss, label="train_loss")
plt.plot(val_loss, label="test_loss")
plt.title("Loss")
plt.legend()
plt.show()
