# -*- coding: utf-8 -*-
import os
import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.slim as slim



class resnet():
    def __init__(self, ):
        pass
    
    def full_connet(self, ):

        w = tf.get_variable("weights-end", shape=[self.layer_list[-1], self.nclass],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable("biases-end", shape=[self.nclass], initializer=tf.constant_initializer(0.0))

        out = tf.matmul(last_out, w) + b
        pass

x = tf.placeholder(tf.float32, [None, 784])
# w表示每一个特征值（像素点）会影响结果的权重
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
# 是图片实际对应的值
  y_ = tf.placeholder(tf.float32, [None, 10])<br>
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # mnist.train 训练数据
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
  #取得y得最大概率对应的数组索引来和y_的数组索引对比，如果索引相同，则表示预测正确
  correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                       y_: mnist.test.labels}))

if __name__ == "__main__":
    main()


