import numpy as np
import cv2
import tensorflow as tf

#加载训练好的模型
def load_model(sess,meta_path,model_path):
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(meta_path)
    saver.restore(sess=sess, save_path=model_path)
    input_x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    pre_softmax_ = tf.get_default_graph().get_tensor_by_name('Softmax:0')
    return input_x,pre_softmax_

#形态学处理
def Morphological_processing(foreground):
    com_list = []
    foreground = cv2.erode(foreground, np.ones((3, 3), np.uint8))
    foreground = cv2.erode(foreground, np.ones((3, 3), np.uint8))
    # 第一次搜索边框，如果小于某一个阈值，将前景区域内边框内填充为黑色
    _, contours, hi = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h <= 200:
            foreground[y:y + h, x:x + w] = 0
    # 第二次搜索边框，填补前景空洞
    _1, contours_1, hi_ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours_1:
        com = cv2.approxPolyDP(contour, 2, closed=True)
        com_list.append(com)
    cv2.drawContours(foreground, com_list, -1, (255, 255, 255), thickness=cv2.FILLED, lineType=8)
    foreground = cv2.dilate(foreground, np.ones((5, 5), np.uint8))

    return foreground

#添加文本
def put_text(frame,target_name,score,x,y):
    if y>=0 and y<=10:
        y = y+20
    cv2.putText(frame, target_name + ':' + str(score),(x, y - 3),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1)

#扩大边框
def get_bigger_contour(x,y,w,h,width):
    if x-width>=0:
        x = x-width
    else:
        x = 0
    if y-width>=0:
        y = y - width
    else:
        y = 0
    if x+w+2*width <=799:
        w = w+2*width
    else:
        w = 799-x
    if y+h+2*width<=575:
        h = h+2*width
    else:
        h = 575-h
    return x,y,w,h