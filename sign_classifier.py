import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import random
import matplotlib.pyplot as plt
import data_preproc as preproc

EPOCHS = 10
BATCH_SIZE = 128
DATASET = preproc.Dataset()


def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6),
                                              mean=mu,
                                              stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,
                         conv1_W,
                         strides=[1, 1, 1, 1],
                         padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16),
                                              mean=mu,
                                              stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,
                         conv2_W,
                         strides=[1, 1, 1, 1],
                         padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120),
                                            mean=mu,
                                            stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84),
                                            mean=mu,
                                            stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 10),
                                            mean=mu,
                                            stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


if __name__ == "__main__":
    normalized = DATASET.pre_process()
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 10)

    # rate = 0.001
    # logits = LeNet(x)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    # loss_operation = tf.reduce_mean(cross_entropy)
    # optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    # training_operation = optimizer.minimize(loss_operation)
    # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    # accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # saver = tf.train.Saver()

    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     num_examples = len(DATASET.X_train)
    #     
    #     print("Training...")
    #     print()
    #     for i in range(EPOCHS):
    #         # X_train, y_train = shuffle(X_train, y_train)
    #         for offset in range(0, num_examples, BATCH_SIZE):
    #             end = offset + BATCH_SIZE
    #             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
    #             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
    #             
    #         validation_accuracy = evaluate(X_validation, y_validation)
    #         print("EPOCH {} ...".format(i+1))
    #         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    #         print()
    #         
    #     saver.save(sess, './lenet')
    #     print("Model saved")

    # with tf.Session() as sess:
    #     saver.restore(sess, tf.train.latest_checkpoint('.'))

    #     test_accuracy = evaluate(X_test, y_test)
    #     print("Test Accuracy = {:.3f}".format(test_accuracy))

    # import pdb;pdb.set_trace()
