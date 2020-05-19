import tensorflow as tf
import numpy as np
import cv2
import os
import os.path as ops

TRAINING = tf.Variable(initial_value=False, dtype=tf.bool, trainable=False)

def identity_block(X_input, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                 kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),
                                 padding='SAME', name=conv_name_base+'2b')
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1),name=conv_name_base+'2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride = 2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,
                                 kernel_size=(1, 1),
                                 strides=(stride, stride),                                 
                                 name=conv_name_base+'2a')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base+'2a', training=TRAINING)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',padding='SAME')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=TRAINING)
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
        x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=TRAINING)

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1),
                                      strides=(stride, stride), name=conv_name_base + '1')
        X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=TRAINING)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result

def global_avg_pooling(x):

    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=False)

    return gap
    
#aux classification, tiny resnet
def aux_classification(X,classes=2):
    # stage 1
    x = tf.layers.conv2d(X, filters=32, kernel_size=(3, 3), strides=(2, 2), name='aux_conv1')
    x = tf.layers.batch_normalization(x, axis=3, name='aux_bn_conv1')
    x = tf.nn.relu(x)
    
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3),strides=(2, 2))
    
    # stage 2
    x = convolutional_block(x, kernel_size=3, filters=[32, 32, 64], stage=2, block='aux_a', stride=1)
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='aux_b')

    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[64, 64, 128], stage=3, block='aux_a', stride=2)
    x = identity_block(x, 3, [64,64,128], stage=3, block='aux_b')

    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[128, 128, 256], stage=4, block='aux_a', stride=2)
    x = identity_block(x, 3, [128, 128, 256], stage=4, block='aux_b')
 
    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[256, 256, 512], stage=5, block='aux_a', stride=2)
    x = identity_block(x, 3, [256, 256, 512], stage=5, block='aux_b')

    x = tf.layers.average_pooling2d(x, pool_size=(7, 7), strides=(1,1))

    flatten = tf.contrib.layers.flatten(x)
    dense1 = tf.layers.dense(flatten, units=64, activation=tf.nn.relu)
    logits = tf.layers.dense(dense1, units=classes, activation=None)
    return logits

'''
#input X should be normalized into[-1,1]
#func: tiny resnet18,begin with 224
#return:  aux,CUEMAP,C0,C1,C2,C3,C4---->
#aux 224x224x3 for classification network with softmax; 
#CUEMAP 224x224x3 for L1 loss when spoof
#C0 112x112x64 for global average pooling ,then softmax(baidu: triplet loss)
#C1 56x56x128 for global average pooling, then softmax(baidu: triplet loss)
#C2 28x28x256 for global average pooling,then softmax(baidu:triplet loss)
#C3 14x14x512 for global average pooling,then softmax(baidu:triplet loss)
#C4 7x7x512 for global average pooling,then softmax(baidu:triplet loss)
'''
def ResNet18_antispoofing(X,classes=2):
    """
    Implementation of the popular ResNet18 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:

    Returns:
    """
    print(X.shape)
    
    x_input=X

    # stage 1
    x = tf.layers.conv2d(X, filters=32, kernel_size=(3, 3), strides=(2, 2),padding='SAME', name='conv1')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
    x = tf.nn.relu(x)
    print(x.shape)
    A0=x
    
    # stage 2
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3),strides=(2, 2),padding='SAME')        
    x = convolutional_block(x, kernel_size=3, filters=[32, 32, 64], stage=2, block='a', stride=1)
    x = identity_block(x, 3, [32, 32, 64], stage=2, block='b')
    A1=x
    print(x.shape)
    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[64,64,128], stage=3, block='a', stride=2)
    x = identity_block(x, 3, [64,64,128], stage=3, block='b')
    A2=x
    print(x.shape)
    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[128, 128, 256], stage=4, block='a', stride=2)
    x = identity_block(x, 3, [128, 128, 256], stage=4, block='b')
    A3=x
    print(x.shape)
    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[256, 256, 512], stage=5, block='a', stride=2)
    x = identity_block(x, 3, [256, 256, 512], stage=5, block='b')
    C4=x
    print(x.shape)
    
    input_shape=tf.shape(x)
    x=tf.image.resize_nearest_neighbor(x,(input_shape[1]*2,input_shape[2]*2))
    x = tf.layers.conv2d(x, filters=256, kernel_size=(2, 2), strides=(1, 1), padding='SAME', name='conv_up0')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv_up0')
    x = tf.nn.relu(x)
    B3=x
    x=tf.concat([B3, A3], axis=-1)
    x= identity_block(x, 3, [512, 512, 512], stage=6, block='a')
    C3=x
    print(x.shape)
    input_shape=tf.shape(x)
    x=tf.image.resize_nearest_neighbor(x,(input_shape[1]*2,input_shape[2]*2))
    x = tf.layers.conv2d(x, filters=128, kernel_size=(2, 2), strides=(1, 1),padding='SAME', name='conv_up1')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv_up1')
    x = tf.nn.relu(x)
    B2=x
    x=tf.concat([B2, A2], axis=-1)
    x= identity_block(x, 3, [256, 256, 256], stage=7, block='a')
    C2=x
    print(x.shape)
    input_shape=tf.shape(x)
    x=tf.image.resize_nearest_neighbor(x,(input_shape[1]*2,input_shape[2]*2))
    x = tf.layers.conv2d(x, filters=64, kernel_size=(2, 2), strides=(1, 1), padding='SAME',name='conv_up2')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv_up2')
    x = tf.nn.relu(x)    
    B1=x
    x=tf.concat([B1, A1], axis=-1)
    x= identity_block(x, 3, [128, 128, 128], stage=8, block='a')
    C1=x
    print(x.shape)
    input_shape=tf.shape(x)
    x=tf.image.resize_nearest_neighbor(x,(input_shape[1]*2,input_shape[2]*2))
    x = tf.layers.conv2d(x, filters=32, kernel_size=(2, 2), strides=(1, 1), padding='SAME',name='conv_up3')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv_up3')
    x = tf.nn.relu(x)     
    B0=x
    x=tf.concat([B0, A0], axis=-1)
    x= identity_block(x, 3, [64, 64, 64], stage=9, block='a')
    C0=x
    print(x.shape)
    input_shape=tf.shape(x)
    x=tf.image.resize_nearest_neighbor(x,(input_shape[1]*2,input_shape[2]*2))
    x = tf.layers.conv2d(x, filters=3, kernel_size=(2, 2), strides=(1, 1), padding='SAME',name='conv_up4')
    x = tf.layers.batch_normalization(x, axis=3, name='bn_conv_up4')    
    x = tf.nn.tanh(x)
    CUEMAP=x
    print(x.shape)
    
    #residual add input with output of cuemap, for further aux classification
    x_shortcut=tf.add(x,x_input)    
    auxin = tf.nn.relu(x_shortcut)
    
    #aux classification subnetwork,still using a similar resnet tiny network
    auxout= aux_classification(auxin,classes)
  
    return auxout,CUEMAP,C0,C1,C2,C3,C4

