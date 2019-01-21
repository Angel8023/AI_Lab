import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
#读取图片
src = cv.imread("lena.jpg",0)
#利用matplotlib显示图片
#dst = cv.GaussianBlur(src, (5,5), 15)
#dst = cv.Sobel(src, -1,1,0)#找边缘
#dst = cv.medianBlur(src,5)
dst = cv.Canny(src, 100, 300, None, 3, True)#用的高斯滤波找边缘
#dst = cv.bilateralFilter(src,0,100,15)#高斯双边滤波
#dst = cv.pyrMeanShiftFiltering(src=src, sp=15, sr=20)#均值偏移滤波
#t,dst = cv.threshold(src,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)#全局模糊
#cv.adaptiveTheashold(src,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,)#自适应模糊
cv.imshow("blur", dst)
cv.waitKey(0)
cv.destroyAllwindows()