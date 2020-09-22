import tensorflow as tf
import numpy as np
from sklearn import datasets
from matplotlib import pyplot
data = datasets.load_iris().data
target = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(target)
#print(target,data)
#tf.random.set_seed(118)

x_train = data[:-30]
y_train = target[:-30]
x_test = data[-30:]
y_test = target[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(3)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(3)

w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

#print(w1)
#print(b1)

lr = 0.05
loss_result = []
test_acc = []
epoch = 50
loss_all = 0

for ep in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        #print(step,x_train,y_train)
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            loss = -tf.reduce_sum(y_*tf.math.log(tf.clip_by_value(y,1e-10,1.0)))
            #loss = tf.reduce_mean(tf.square(y-y_))
            loss_all += loss
        grads = tape.gradient(loss, [w1, b1])
        #print(loss)
        #print(y_)
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    print("epoch {}, loss {}".format(ep, loss_all/40))
    loss_result.append(loss_all/40)
    loss_all=0

    total_correct, total_number = 0,0
    for x_test, y_test in test_db:
        #print(y_test)
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        #print(y)
        pred = tf.argmax(y, axis=1)
        #print(pred)
        pred = tf.cast(pred, y_test.dtype)
        #print(pred)
        correct = tf.cast(tf.equal(pred,y_test), dtype=tf.int32)
        #print(correct)
        correct = tf.reduce_sum(correct)
        #print(int(correct))
        total_correct += int(correct)
        total_number += y_test.shape[0]
    acc = total_correct/total_number
    test_acc.append(acc)
    print("acc:",acc)
pyplot.title("acc curve")
pyplot.xlabel("epoch")
pyplot.ylabel("acc")
pyplot.plot(test_acc,label="Accuracy")
pyplot.legend()
pyplot.show()

pyplot.title("loss curve")
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.plot(loss_result,label="loss")
pyplot.legend()
pyplot.show()