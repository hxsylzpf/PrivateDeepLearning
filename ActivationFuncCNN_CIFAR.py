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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
from math import sqrt;

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.models.image.cifar10 import cifar10;

FLAGS = tf.app.flags.FLAGS;

D = 50000; #Data size#
numHidUnits = 12*12; #Number of hidden units in one convolution layer#
numFeatures = 16*16*3; #Number of features in one convolution layer#
Delta = 2*numHidUnits*numFeatures; #Function sensitivity#
epsilon = 1.0; #Privacy budget epsilon#
loc, scale1 = 0., Delta*numHidUnits/(epsilon*255*FLAGS.batch_size); #0-mean and variant of noise#
W_conv1Noise = np.random.laplace(loc, scale1, 128*24*24*3); #This is the latest version of W_conv1Noise#
W_conv1Noise = np.reshape(W_conv1Noise, [128, 24, 24, 3]); #This is the latest version of W_conv1Noise#

def inference(images):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = cifar10._variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(images + W_conv1Noise, kernel, [1, 1, 1, 1], padding='SAME') #Perturb the first affine transformation layer#
    #conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')#noiseless model
    biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.0)) #Define biases#
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    cifar10._activation_summary(conv1)
    
  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  
  # local response normalization (LRN): norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = cifar10._variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = cifar10._variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    cifar10._activation_summary(conv2)

  # local response normalization (LRN): norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
    dim = reshape.get_shape()[1].value
    weights = cifar10._variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = cifar10._variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    cifar10._activation_summary(local3)
    
  # local4
  with tf.variable_scope('local4') as scope:
    weights = cifar10._variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = cifar10._variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    cifar10._activation_summary(local4)
    
#print(images.get_shape());
#print(norm1.get_shape());
#print(pool2.get_shape());
#print(local3.get_shape());
#print(local4.get_shape());

  # linear layer(WX + b),
  # We don't apply softmax here because 
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits 
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = cifar10._variable_with_weight_decay('weights', [192, 10],
                                          stddev=1/192.0, wd=0.0)
    biases = cifar10._variable_on_cpu('biases', [10],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    cifar10._activation_summary(softmax_linear)
  return softmax_linear
  
def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    images, labels = cifar10.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter('/tmp/cifar10_train', sess.graph)
    print (FLAGS.batch_size);
    for step in xrange(100000):#100000
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 100 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 100 == 0:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically.
      if step % 500 == 0 or (step + 1) == 1000000:
        checkpoint_path = os.path.join('/tmp/cifar10_train', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step);


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract();
  if tf.gfile.Exists('/tmp/cifar10_train'):
    tf.gfile.DeleteRecursively('/tmp/cifar10_train');
  tf.gfile.MakeDirs('/tmp/cifar10_train');
  print(scale1);
  print(scale2);
  print(scale3);
  train()


if __name__ == '__main__':
  tf.app.run()
