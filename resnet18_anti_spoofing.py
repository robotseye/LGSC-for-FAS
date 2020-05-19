import tensorflow as tf
import numpy as np
import cv2
import os
import os.path as ops

TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

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

#parse dataset list
def parse_imagenet_datasets(path):
    file_open=open(path)
    filelist=[]
    for aline in file_open:
        filelist.append(aline)
    allcount=len(filelist)
    X_train=[]
    Y_train=[]
    for i in range(allcount):
        imgname,label=filelist[i].split(' ')  
        X_train.append(imgname)
        Y_train.append(int(label))

    return X_train,Y_train
    
#define dataset method to fetch,sub mean and normalize to [-1,1]
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded,tf.convert_to_tensor([224, 224], dtype=tf.int32))
    image_resized = tf.image.random_flip_left_right(image_resized)
    image_resized = tf.subtract(image_resized,127)
    image_resized = tf.div(image_resized,128)
    return image_resized, label
    
#fix batch=16, first 8 is live and other is spoofing
def main():
    
    #imagelist to train
    train_live_path='./train_live_list.txt'
    train_spoof_path='./train_spoof_list.txt'    
    test_path='./test_list.txt'
    
    global TRAINING
   
    classes = 2
    H_size  = 224
    W_size  = 224
    C_size  = 3
    
    #parse train data with 16 live + 16 spoof ->32 batch
    X_train_live,Y_train_live = parse_imagenet_datasets(train_live_path)
    X_train_spoof,Y_train_spoof = parse_imagenet_datasets(train_spoof_path)
    
    X_train_live_tensor=tf.convert_to_tensor(X_train_live,dtype=tf.string)
    Y_train_live_tensor=tf.convert_to_tensor(Y_train_live,dtype=tf.int32)
    X_train_spoof_tensor=tf.convert_to_tensor(X_train_spoof,dtype=tf.string)
    Y_train_spoof_tensor=tf.convert_to_tensor(Y_train_spoof,dtype=tf.int32)
    
    train_live_dataset = tf.data.Dataset.from_tensor_slices((X_train_live_tensor, Y_train_live_tensor)) #contrib for 1.3 of tensorflow
    train_live_dataset = train_live_dataset.map(_parse_function)
    train_live_dataset = train_live_dataset.prefetch(buffer_size=100)
    train_live_dataset = train_live_dataset.shuffle(buffer_size=10).batch(8).repeat(50)
    train_live_iterator=train_live_dataset.make_one_shot_iterator()
    train_one_batch_live=train_live_iterator.get_next()
    
    train_spoof_dataset = tf.data.Dataset.from_tensor_slices((X_train_spoof_tensor, Y_train_spoof_tensor)) #contrib for 1.3 of tensorflow
    train_spoof_dataset = train_spoof_dataset.map(_parse_function)
    train_spoof_dataset = train_spoof_dataset.prefetch(buffer_size=100)
    train_spoof_dataset = train_spoof_dataset.shuffle(buffer_size=10).batch(8).repeat(50)
    train_spoof_iterator=train_spoof_dataset.make_one_shot_iterator()
    train_one_batch_spoof=train_spoof_iterator.get_next()
    
    #parse test data with 16 batch,mix label
    X_test,Y_test= parse_imagenet_datasets(test_path)
    X_test_tensor=tf.convert_to_tensor(X_test,dtype=tf.string)
    Y_test_tensor=tf.convert_to_tensor(Y_test,dtype=tf.int32)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_tensor, Y_test_tensor)) 
    test_dataset = test_dataset.map(_parse_function)
    test_dataset = test_dataset.prefetch(buffer_size=100)
    test_dataset = test_dataset.shuffle(buffer_size=100).batch(4).repeat(20)
    test_iterator=test_dataset.make_one_shot_iterator()
    test_one_batch=test_iterator.get_next()        

    learningrate = tf.placeholder(tf.float32, shape=[])
    X = tf.placeholder(tf.float32, shape=(None, H_size, W_size, C_size), name='X')
    Y = tf.placeholder(tf.int32, shape=(None,), name='Y')
    
    print("debug,begin of resnet18 anti-spoofing network building")
    
    aux,CUEMAP,C0,C1,C2,C3,C4 = ResNet18_antispoofing(X,classes)
    
    #regular L2
    reg_l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-5), tf.trainable_variables())
    
    loss_aux = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=aux))
    
    CUEMAP_HEAD,CUEMAP_BOTTOM=tf.split(CUEMAP,2,axis=0)
    cue_L2_loss=tf.nn.l2_loss(CUEMAP_HEAD)
    
    D0=global_avg_pooling(C0)
    logits_D0 = tf.layers.dense(D0, units=classes, activation=None)
    loss_D0 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits_D0))
    
    D1=global_avg_pooling(C1)
    logits_D1 = tf.layers.dense(D1, units=classes, activation=None)
    loss_D1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits_D1))

    D2=global_avg_pooling(C2)
    logits_D2 = tf.layers.dense(D2, units=classes, activation=None)
    loss_D2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits_D2))

    D3=global_avg_pooling(C3)
    logits_D3 = tf.layers.dense(D3, units=classes, activation=None)
    loss_D3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits_D3))

    D4=global_avg_pooling(C4)
    logits_D4 = tf.layers.dense(D4, units=classes, activation=None)
    loss_D4 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=logits_D4))        
    #step0 loss= 2*loss_aux + 0.00001*cue_L2_loss +0.5*(loss_D0+loss_D1+loss_D2+loss_D3+loss_D4)+0.5*reg_l2 
    loss= 2*loss_aux + 0.00001*cue_L2_loss +0.5*(loss_D0+loss_D1+loss_D2+loss_D3+loss_D4)+0.5*reg_l2
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningrate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    
     # accuracy    
    correct = tf.nn.in_top_k(aux, Y, k=1)    
    correct = tf.cast(correct, tf.float16)    
    accuracy = tf.reduce_mean(correct)    
       
   
    saver = tf.train.Saver()
    model_save_dir = 'model/resnet_antispoof'
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    fd = open("train_antispoofing_log.txt", "w")
    print("debug,begin of network session run")
    weights_path=model_save_dir
    module_file=tf.train.latest_checkpoint(weights_path)
    print("loading ",module_file)
    
    print("debug,begin of network session run")    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path=module_file)
        _,steprestore= module_file.split('-')
        step=int(steprestore)
        print('continue with step=',step)
        #step=0
        #get data   
        
        lr=0.0001
        sess.run(tf.assign(TRAINING, True))
        for i in range(50000):
            trainbatch_live=sess.run(train_one_batch_live)
            trainbatch_spoof=sess.run(train_one_batch_spoof)
            x_train=np.concatenate((trainbatch_live[0],trainbatch_spoof[0]),axis=0)
            y_train=np.concatenate((trainbatch_live[1],trainbatch_spoof[1]),axis=0)

            _, cost_sess,coss_aux,cue_L2_coss,coss_D0,coss_D1,coss_D2,coss_D3,coss_D4,train_acc = sess.run([train_op, loss, loss_aux,cue_L2_loss,loss_D0,loss_D1,loss_D2,loss_D3,loss_D4,accuracy], feed_dict={learningrate:lr, X: x_train, Y: y_train})
            step=step+1
            
            if (step%10)==0:
                fd.write('step='+str(step)+' loss='+str(cost_sess)+' \n')
                fd.flush()
            if step>0 and step<2000:
                lr=0.002*step/2000
            if step>2000 and step<10000: 
                lr=0.002   
            if step>10000 and step<20000:
                lr=0.001
            if step>20000 and step<30000:
                lr=0.0001
            if step>30000:
                lr=0.00001
            if step % 100 == 0:
                print("step=%d,loss=%f,loss_aux=%f,cue_L2_loss=%f,loss_D0=%f,loss_D1=%f,loss_D2=%f,loss_D3=%f,loss_D4=%f,train_acc=%f"%(step,cost_sess,coss_aux,cue_L2_coss,coss_D0,coss_D1,coss_D2,coss_D3,coss_D4,train_acc))
            
            if(step>100 and step%1000==0):      
                checkpoint_path = os.path.join(model_save_dir, 'model_baidu_antispoofing_v1.ckpt')                
                saver.save(sess, checkpoint_path, global_step=step)          
                sess.run(tf.assign(TRAINING, False))
                testing_acur=0
                testcount=0
                try:
                    while True:
                        testbatch=sess.run(test_one_batch)   
                        testing_acur = testing_acur + sess.run(accuracy, feed_dict={X: testbatch[0], Y: testbatch[1]})
                        testcount=testcount+1
                        #enough for all test fetched
                        if testcount>200: #one  1/4 epoch 780x4=all test pics
                            fd.write("testing acurracy: "+ str( testing_acur/testcount)+' \n')
                            print("testing acurracy: "+ str( testing_acur/testcount))
                            break
                except tf.errors.OutOfRangeError:
                    print("test done! ")
                sess.run(tf.assign(TRAINING, True))
        fd.close()
if __name__ == '__main__':
    main()


