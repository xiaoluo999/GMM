'''

近期在使用opencv_python分析视频过程中总是遇到各种问题，使用opencv_python操作视频处理类的需求时总是遇到has no attribute 'bgsegm'等问题。
从网上找了但缺少完整的说明，故特地说明下。
遇到has no attribute 'bgsegm'等问题，说明安装的opencv_python版本需要更新了，或者没有安装contrib包。
详细如下：在opencv3.0以后的版本中，只有createBackgroundSubtractorKNN和createBackgroundSubtractorMOG2函数，而createBackgroundSubtractorGMG与createBackgroundSubtractorMOG被移动到opencv_contrib包中了。
故使用上要如下方式：
cv2.createBackgroundSubtractorKNN([,history[, dist2Threshold[, detectShadows]]])
cv2.createBackgroundSubtractorMOG2([,history[, varThreshold[, detectShadows]]])
要调用createBackgroundSubtractorGMG与createBackgroundSubtractorMOG则采用如下：
cv2.bgsegm.createBackgroundSubtractorGMG([,initializationFrames[, decisionThreshold]])
cv2.bgsegm.createBackgroundSubtractorMOG([,history[, nmixtures[, backgroundRatio[,noiseSigma]]]])
当出现上述问题时，则说明你的python中没有安装contrib包。
       
安装contrib包时有如下3种方式：

1、可从https://pypi.python.org/pypi/opencv-contrib-python去下载并安装,
   小罗本实验安装包：opencv_contrib_python-3.3.1.11-cp35-cp35m-win_amd64.whl (38.4 MB)

2、到https://www.lfd.uci.edu/~gohlke/pythonlibs/去下载opencv_python‑3.3.1+contrib‑cp36‑cp36m‑win_amd64.whl。

3、contrib包的源代码地址：https://github.com/opencv/opencv_contrib也可自行编译。

'''
import numpy as np
import cv2
import time
import datetime
cap = cv2.VideoCapture(r"F:\data\fire\93.mp4")
#cap = cv2.VideoCapture(0)

#高斯混合模型，使用一种通过K高斯分布的混合来对每个背景像素进行建模的方法(K = 3〜5）。
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=5,backgroundRatio=0.7)

#改进的高斯混合模型，一个重要特征是 它为每个像素选择适当数量的高斯分布，它可以更好地适应不同场景的照明变化等。噪音点为啥这么多
#fgbg = cv2.createBackgroundSubtractorMOG2()
count = 0
while (1):
    count+=1
    ret, frame = cap.read()
    if not isinstance(frame,np.ndarray):
        break
    width_resize = 500
    height_resize = 500
    x_scale = frame.shape[1]*1.0/width_resize
    y_scale = frame.shape[0] * 1.0 / height_resize
    src = frame.copy()
    frame = cv2.resize(frame,dsize=(500,500))
    fgmask = fgbg.apply(frame,0.005)
    #开运算去除噪音点
    elem = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,ksize=(3,3))
    #fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel=elem)
    #闭运算去除空洞
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(15, 15))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel=elem)

    _,cnts,_ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maxArea = 10
    list_win_name = []
    for i,c in enumerate(cnts):
        Area = cv2.contourArea(c)
        (left,top,width,height) = cv2.boundingRect(c)
        if Area < maxArea or width<5 or height<5:
            fgmask[top:top+height,left:left+width] = 0
            (x, y, w, h) = (0, 0, 0, 0)
            continue
        x = int(left * x_scale)
        w = int(width * x_scale)
        y = int(top * y_scale)
        h = int(height * y_scale)
        crop = src[y:y+h,x:x+w]
        win_name = "%d_crop_%d_%d_%d_%d.jpg"%(count,x,y,w,h)
        list_win_name.append(win_name)
        cv2.imshow(win_name,crop)
        cv2.imwrite(win_name,crop)
    cv2.imshow('frame', frame)
    cv2.imshow("mask",fgmask)
    k = cv2.waitKey(10) & 0xff
    for name in list_win_name:
        cv2.destroyWindow(name)
    if k == 27:
        break

