import numpy as np
import tensorflow as tf


def model(features, labels, mode):
    W = tf.get_variable("W", [1], dtype=tf.float64)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features['x'] + b
    loss = tf.reduce_sum(tf.square(y - labels))

    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(
               mode=mode, predictions=y, loss=loss, train_op=train)


estimator = tf.contrib.learn.Estimator(model_fn=model)
# define dataset
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_test = np.array([2., 5., 8., 1.])
y_test = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
test_input_fn = tf.contrib.learn.io.numpy_input_fn(
        {"x": x_test}, y_test, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate
train_metrics = estimator.evaluate(input_fn=train_input_fn)
test_metrics = estimator.evaluate(input_fn=test_input_fn)
print("train metrics: %r" % train_metrics)
print("test metrics: %r" % test_metrics)
