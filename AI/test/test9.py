import tensorflow as tf
import numpy as np
from sklearn import datasets
from matplotlib import pyplot
data = [[7,26,6,60],[1,29,15,52],[11,56,8,20],[11,31,8,47],[7,52,6,33],[11,55,9,22],[3,71,17,6],[1,31,22,44],[2,54,18,22],[21,47,4,26],[1,40,23,34],[11,66,9,12],[10,68,8,12]]
target = [78.5,74.3,104.3,87.6,95.9,109.2,102.7,72.5,93.1,115.9,83.8,113.3,109.4]

np.random.seed(116)
np.random.shuffle(data)
np.random.seed(116)
np.random.shuffle(target)
tf.random.set_seed(118)
data = np.array(data)
target = np.array(np.reshape(target,(-1,1)))
#print(target)
target = tf.cast(target, tf.float32)
num_data = len(data)
# print(data)
#print(target)
data = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
#print(data)

b1 = np.ones(num_data).reshape(-1,1)
#print(b1)
data = tf.cast(tf.concat([b1, data], axis=1),tf.float32)
#print(data)

train_db = tf.data.Dataset.from_tensor_slices((data,target)).batch(5)
#test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(3)

w1 = tf.Variable(tf.random.truncated_normal([5,1], stddev=0.1, seed=1))
#b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

#print(w1)
#print(b1)

lr = 0.05
loss_result = []
test_result = []
true_result = []
epoch = 500
loss_all = 0

for ep in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        #print(step,x_train,y_train)
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1)
            #y = tf.nn.softmax(y)
            #y_ = tf.one_hot(y_train, depth=3)
            #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            #loss = -tf.reduce_sum(y_*tf.math.log(tf.clip_by_value(y,1e-10,1.0)))
            loss = tf.reduce_mean(tf.square(y-y_train))
            loss_all += loss
        grads = tape.gradient(loss, w1)
        #print(loss)
        #print(y_)
        w1.assign_sub(lr*grads)
        #b1.assign_sub(lr*grads[1])
    print("epoch {}, loss {}".format(ep, loss_all/3))
    loss_result.append(loss_all/3)
    loss_all=0

for x_test, y_test in train_db:
    #print(y_test)
    y = tf.matmul(x_test, w1)
    #pred = tf.cast(y, y_test.dtype)
    #print("pred", pred)
    #pred = tf.cast(y, tf.float32)
    for i in y.numpy().flatten():
        test_result.append(i)
    for i in y_test.numpy().flatten():
        true_result.append(i)

#test_result = tf.cast(test_result,tf.float32)
print(test_result)
print(true_result)
pyplot.subplot(121)
pyplot.title("pred and true curve")
pyplot.xlabel("number")
pyplot.ylabel("pred and true")
pyplot.plot(test_result,label="Pred")
pyplot.plot(true_result,label="True")
pyplot.legend()

pyplot.subplot(122)
pyplot.title("loss curve")
pyplot.xlabel("epoch")
pyplot.ylabel("loss")
pyplot.plot(loss_result,label="loss")
pyplot.legend()
pyplot.show()
