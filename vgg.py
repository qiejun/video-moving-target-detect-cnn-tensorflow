import tensorflow as tf
import numpy as np

"""
定义vgg16
"""

class VGG16(object):

    def __init__(self):
        self.vgg_dict = np.load('./vgg16/vgg16.npy',encoding='latin1').item()
        self.input = tf.placeholder(tf.float32,[None,224,224,3])

        self.conv1_1 = self.conv_layer(self.input,'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1,'conv1_2')
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2,2,2,'same',name='pool1')

        self.conv2_1 = self.conv_layer(self.max_pool1, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2,2,2,'same',name='pool2')

        self.conv3_1 = self.conv_layer(self.max_pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_3,2,2,'same',name='pool3')

        self.conv4_1 = self.conv_layer(self.max_pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.max_pool4 = tf.layers.max_pooling2d(self.conv4_3,2,2,'same',name='pool4')

        self.conv5_1 = self.conv_layer(self.max_pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.max_pool5 = tf.layers.max_pooling2d(self.conv5_3, 2, 2, 'same', name='pool5')

        #全连接层参数量太大，加载的时候比较慢，且fine-tuning时只利用了卷积层，所以将fc层全部注释
        # self.fc6 = self.fc_layer(self.max_pool5,'fc6')
        # self.relu6 = tf.nn.relu(self.fc6)
        # self.fc7 = self.fc_layer(self.fc6,'fc7')
        # self.relu7 = tf.nn.relu(self.fc7)
        # self.fc8 = self.fc_layer(self.relu7,'fc8')
        # self.softmax = tf.nn.softmax(self.fc8)

    def conv_layer(self,input,layer_name):
        with tf.variable_scope(layer_name):
            weight = self.vgg_dict[layer_name][0]
            bias = self.vgg_dict[layer_name][1]
            conv = tf.nn.conv2d(input,weight,[1,1,1,1],padding='SAME')
            output = tf.nn.relu(conv+bias)
            return output

    def fc_layer(self,input,layer_name):
        with tf.variable_scope(layer_name):
            shape = input.get_shape()
            if len(shape.as_list())==4:
                input = tf.reshape(input,(-1,shape[1]*shape[2]*shape[3]))
            weight = self.vgg_dict[layer_name][0]
            bias = self.vgg_dict[layer_name][1]
            fc = tf.matmul(input,weight)
            output = fc+bias
            return output
