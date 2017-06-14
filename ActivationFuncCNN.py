# Copyright 2017 Hai Phan, NJIT. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np;
import tensorflow as tf;
from tensorflow.python.framework import ops;
from tensorflow.examples.tutorials.mnist import input_data;
import argparse;
import pickle;
from datetime import datetime
import time
FLAGS = None;

#Initalize random weights with a given shape#
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1);
  return tf.Variable(initial);

#Initalize random a bias with a given shape#
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape);
  return tf.Variable(initial);

#Create a convolution layer: x is inputs, W is parameters#
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME');

#Create a max-pooling layer: x is inputs#
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME');

def main(_):
  D = 50000; #Data size#
  numHidUnits = 14*14; #Number of hidden units in one convolution layer#
  numFeatures = 18*18; #Number of features in one convolution layer#
  Delta = 2*numHidUnits*numFeatures; #Function sensitivity#
  epsilon = 0.5; #Privacy budget epsilon#
  
  loc, scale1 = 0., Delta/(epsilon*D); #0-mean and variant of noise#
  
  W_conv1Noise = np.random.laplace(loc, scale1, 28 * 28);#This is the latest version of W_conv1Noise#
  W_conv1Noise = np.reshape(W_conv1Noise, [-1, 28, 28, 1]);#This is the latest version of W_conv1Noise#
  
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True);

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784]);
  x_image = tf.reshape(x, [-1,28,28,1]);
  W_conv1 = weight_variable([5, 5, 1, 32]);
  b_conv1 = bias_variable([32]);
  h_conv1 = tf.nn.relu(conv2d(x_image + W_conv1Noise, W_conv1) + b_conv1); #Perturb the first affine transformation layer#
  #h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1);#noiseless model#
  h_pool1 = max_pool_2x2(h_conv1);
  
  #Second convolution layer#
  W_conv2 = weight_variable([5, 5, 32, 64]);
  b_conv2 = bias_variable([64]);
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2);
  h_pool2 = max_pool_2x2(h_conv2); #max-pooling layer#
  
  #Fully connected layer 1#
  W_fc1 = weight_variable([7 * 7 * 64, 1024]);
  b_fc1 = bias_variable([1024]);
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]);
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1);
  
  #Dropout with dropout probability keep_prob#
  keep_prob = tf.placeholder(tf.float32);
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob);
  
  #Fully connected layer 2#
  W_fc2 = weight_variable([1024, 10]);
  b_fc2 = bias_variable([10]);
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2;
  
  #Define loss and optimizer#
  y_ = tf.placeholder(tf.float32, [None, 10]);
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv));
  train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy);
  sess = tf.InteractiveSession();
  
  #Define prediction accuracy#
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1));
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));
  
  #Run the Tensorflow session#
  sess.run(tf.initialize_all_variables());
  
  start_time = time.time(); #start time#
  for i in range(88001): #88001 = 160 epochs#
    batch = mnist.train.next_batch(100); #Randomly select a training batch#
    if i%550 == 0: #Report the prediction accuracy on testing data after each training epoch#
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0});
      print("step \t %d \t training accuracy \t %g"%(i, train_accuracy));
      print("step \t %d \t test accuracy \t %g"%(i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})));
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5});
  duration = time.time() - start_time; #duration = endtime - start_time#
  print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})); #Test the model on testing data#
  #print(b_fc2.eval());
  print(float(duration)); #Print out the running time#

if __name__ == '__main__':
  if tf.gfile.Exists('/tmp/mnist_logs'):
    tf.gfile.DeleteRecursively('/tmp/mnist_logs');
  tf.gfile.MakeDirs('/tmp/mnist_logs');
  
  parser = argparse.ArgumentParser();
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data');
  FLAGS = parser.parse_args();
  tf.app.run();
