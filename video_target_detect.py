import cv2
import numpy as np
import tensorflow as tf
import pickle
import utils

"""
视频中运动目标检测：
加载字典找到需要检测目标的key
运行moveing_target_detect()
"""

#加载保存的字典
file = open('class_dict.pickle','rb')
dict = pickle.load(file)
print(dict)
file.close()

def moveing_target_detect(video_file,target_name,meta_path,model_path):
    sess = tf.Session()
    cv2.namedWindow('frame')
    cv2.namedWindow('foreground')
    cv2.moveWindow('frame', 10, 200)
    cv2.moveWindow('foreground', 800, 200)
    input_x, pre_softmax_ = utils.load_model(sess,meta_path=meta_path,model_path=model_path)# 加载训练好的模型
    cap = cv2.VideoCapture(video_file)
    GMM = cv2.createBackgroundSubtractorMOG2(history=5000, varThreshold=10)#混合高斯背景建模
    framenum = 0

    while(1):
        roi_list = []
        index_list = []
        cache_coordinate = []
        ret,frame = cap.read()
        frame=cv2.resize(frame,(800,576))
        GMM_img = GMM.apply(frame,learningRate=-1)#前景提取
        ret1 ,foreground = cv2.threshold(GMM_img,200,255,cv2.THRESH_BINARY)#二值化
        foreground = utils.Morphological_processing(foreground) #形态学处理
        _, contours_filled, hi = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #寻找运动目标对应的原图区域
        for index,contour in enumerate(contours_filled):
            x,y,w,h=cv2.boundingRect(contour)
            if w*h>=50:
                x,y,w,h = utils.get_bigger_contour(x, y, w, h, width=50)
                cache_coordinate.append((x,y,w,h))
                roi = frame[y:y + h, x:x + w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (128, 128))
                roi = np.array(roi) * (1. / 255)
                roi_list.append(roi)
                index_list.append(index)

        #对运动目标预测分类，并框出感兴趣运动目标
        if len(index_list)!=0:
            roi_imgs=np.array(roi_list)
            pre_softmax=sess.run(pre_softmax_,feed_dict={input_x:roi_imgs})
            pre_argmax=np.argmax(pre_softmax,1)
            pre_argmax_list,pre_softmax_list=list(pre_argmax),list(pre_softmax)
            for index,pre in enumerate(pre_argmax_list):
                 if pre==dict[target_name] and pre_softmax_list[index][dict[target_name]] >= 0.8:
                    score = pre_softmax_list[index][dict[target_name]]
                    x,y,w,h = cache_coordinate[index]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    utils.put_text(frame,target_name,score,x,y)

        framenum = framenum + 1
        cv2.imshow('frame',frame)
        cv2.imshow('foreground',foreground)
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows()

moveing_target_detect(video_file='E:\\video_data\\people2.mp4',
                      target_name='people',
                      meta_path='D:\\python\\smoke_detection_4.23\\save\\6.5\\smoke.ckpt2000.meta',
                      model_path='D:\\python\\smoke_detection_4.23\\save\\6.5\\smoke.ckpt2000')