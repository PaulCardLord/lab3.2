import sys

import tensorflow as tf

import loaderMNIST
from userInput import MatrixInput


def main(_):
    mnist = loaderMNIST.mnist

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    for i in range(1000):
        if i % 100 == 0:
            print("iteration# {}: {}".format(i, sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                        y_: mnist.test.labels})))

        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    def classify(matrix):
        prob = sess.run(tf.nn.softmax(y, 1), feed_dict={x: matrix})
        max = sess.run(tf.argmax(y, 1), feed_dict={x: matrix})

        result = prob[0].tolist()
        result.append(max[0].item())
        return result

    MatrixInput(classify)


if __name__ == '__main__':
    tf.app.run(main=main, argv=sys.argv)
