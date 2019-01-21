import cv2 as cv2
import numpy
#调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap=cv2.VideoCapture(0)
while True:
    #从摄像头读取图片
    ret,img=cap.read()
    #转为灰度图片
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #显示摄像头，背景是灰度。
    ret = cap.set(3,640)
    ret = cap.set(4, 480)
    cv2.imshow("img",img)
    #保持画面的持续。
    k=cv2.waitKey(25)#等待多少秒执行，0无限等待，并返回键入值，默认返回-1
    if k == 27:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break
    elif k==ord("s"):
        #通过s键保存图片，并退出。
        cv2.imwrite("image2.jpg",gray)
        cv2.destroyAllWindows()
        break

