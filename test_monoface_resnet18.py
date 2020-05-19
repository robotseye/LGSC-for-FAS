#68 points not necessary just for distance and size judge,as for noise outstanding, disparity is not enough as to baseline<=3
#just using centerface, its landmark and face rects
#add moving detection to reduce NPU processing,and one camera work when no face find
import yaml
import numpy as np
import cv2
import math
import os
import tensorflow as tf
from mobilev2 import conv_blocks as ops
from mobilev2 import mobilenet
from mobilev2 import mobilenet_v2
from centerface import CenterFace
from resnet18_anti_spoofing_network import *

slim = tf.contrib.slim

pixel_size=7.5
baseline=17
f_len=2
disparity_offset=8

DETWIDTH=640
DETHEIGHT=480
num_classes=2
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 

class stereoSpoofingRecognition(object):    
    def __init__(self,modelpath,threshold):
        self.threshold=threshold
        self.module_file=modelpath
        self.face_dic={0:'live',1:'spoof'}
        self.X = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='X') 
        classes=len(self.face_dic)
        aux,CUEMAP,C0,C1,C2,C3,C4 = ResNet18_antispoofing(self.X,classes)
        self.aux=aux
        self.spoofing=tf.nn.softmax(aux,name='Softmax')
        
        self.graph=tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session(config=config)
        
        var_list = tf.trainable_variables()     
        g_list = tf.global_variables()    
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]    
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]        
        var_list += bn_moving_vars            
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)        
        self.sess.run(tf.global_variables_initializer())    
        
        saver.restore(self.sess, modelpath+'model_baidu_antispoofing_v1.ckpt-6000') 
        print('init complete!')
        
                
    def hand_call(self,imgl):
              
        face_ret,aux = self.sess.run([self.spoofing,self.aux],feed_dict={self.X:imgl})         
        print(face_ret,aux)   
        face_ret=face_ret.tolist()[0]
        class_idx=face_ret.index(max(face_ret))
        class_name=self.face_dic[class_idx]
        score=round(max(face_ret),2)
        print('face is '+class_name+' score='+str(score))
        
        return class_name,score

def visible_show(label,img):
    label_size, baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(0, label_size[1])
    left =0
    x1=left;
    x2=left + round(1.5 * label_size[0])
    y1=top - round(1.5 * label_size[1])
    y2=top + baseline
    cv2.rectangle(
        img, 
        (x1, y1),
        (x2,y2), (255, 255, 255),
        cv2.FILLED)

    cv2.putText(img, label, (left, top), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                        0.75, (0, 0, 0), 1)
    return img
def keep_resize(img,width,height):
    h,w,c=img.shape
    if(h==height and w==width):
        return img
    t_w=0
    t_h=0
    if((h/height) < (w/width)):
        t_w=width
        t_h=int(width*h/w)
        dimg=cv2.resize(img,(t_w,t_h))
        cv2.imshow('or',dimg)
        ret=np.empty([height,width,c], dtype=np.uint8)
        print(ret.shape)
        for i in range(t_h):
            ret[i,:,:]=dimg[i,:,:]
        
    else:
        t_h=height
        t_w=int(height*w/h)
        dimg=cv2.resize(img,(t_w,t_h))
        cv2.imshow('or',dimg)
        ret=np.empty([height,width,c], dtype=np.uint8)
        print(ret.shape)
        for i in range(t_w):
            ret[:,i,:]=dimg[:,i,:]
  
    return ret
    
def test_video(centerface,hpr,video0,calibed=True):   
    global pic_count
    #capture
    cap0 = cv2.VideoCapture(video0)
  
    ret =True
    #store for farther trainning
    
    #init facedetect model
    ret0,img0 = cap0.read()

    while(ret):
        ret,img = cap0.read()

        #using dlib to detect and calc disparity to get left and right corresponding face images,prepare for cnn
        img=keep_resize(img,DETWIDTH,DETHEIGHT)
        dets,keypoints = centerface(img, threshold=0.6)
        if len(dets)<1:
            continue
        #only using max one face
        maxidx=0
        maxarea=0
        for i in range(len(dets)):
            area=abs((dets[i][3]-dets[i][1])*(dets[i][2]-dets[i][0]))
            if(maxarea<area):
                maxarea= area
                maxidx=i
            
        det=dets[maxidx]
        keypoint=keypoints[maxidx]
        faceimg=img[int(det[1]):int(det[3]),int(det[0]):int(det[2])]
        
        rightface=faceimg
            
        imgr_s=cv2.resize(rightface,(224,224))
        imgr=imgr_s.reshape(224,224,3)
        imgr=imgr.reshape(1,224,224,3)
        imgr=imgr.astype(np.float32)
        imgin=(imgr-127)/128
        cls,score= hpr.hand_call(imgin)

        #show result
        
        right_result=visible_show(cls,rightface)
        cv2.imshow('rightface',right_result)
        cv2.waitKey(10)
 

def main_camera():
    #network initialize of cnn for stereo faces anti spoofing
    modelpath = "./checkpoint_resnet/"
    hpr=stereoSpoofingRecognition(modelpath,0.7)
    centerface = CenterFace(DETHEIGHT, DETWIDTH)

    test_video(centerface,hpr,0,calibed=True)

main_camera()
