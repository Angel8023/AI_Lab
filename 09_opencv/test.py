#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import serial

# from picamera.array import PiRGBArray
# from picamera import PiCamera
# import time
# import copy
import video
from matplotlib import pyplot as plt

class App(object):
    def __init__(self, color):
        self.cam = video.create_capture(0)
#         self.cam = PiCamera()
#         self.cam.resolution = (320,240)
#         self.cam.framerate = 32
#         self.rCa = PiRGBArray(self.cam, size=(320,240))
#         time.sleep(0.1) 
#        self.cam.capture(self.rCa, format='bgr')
#        self.frame = self.rCa.array
#        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        if color == 0:
            self.roi = cv2.imread( 'hong.jpg' )
            self.flag = "Hong"
        else :
            self.flag = "Lan"
            self.roi = cv2.imread('lan.jpg')
            self.selection = None
            self.tracking_state = 0
            self.show_backproj = False
#         self.ser = serial.Serial('/dev/ttyAMA0',115200,timeout=0.5)

                    
    def start(self):
        self.tracking_state = 0
        #x, y = np.int16([220, 110]) # BUG
        if self.flag == 'Hong':
                self.selection = (4, 6, 407, 304)
        else:
            self.selection = (40, 54, 296, 230)
            self.tracking_state = 1
    #        print "start"

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self):
        roi = self.roi
        self.start()
        while True:
#         for frame in self.cam.capture_continuous(self.rCa, format='bgr', use_video_port=True):
            ret, self.frame = self.cam.read()
#             self.frame = frame.array
            vis = self.frame.copy()
#             vis = copy.deepcopy(self.frame)
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
#             self.selection = 1

            if self.selection:
#                 x0, y0, x1, y1 = 220, 110, 358, 245
                x0, y0, x1, y1 = self.selection
                self.track_window = (x0, y0, x1-x0, y1-y0)
#                 hsv_roi = hsv[y0:y1, x0:x1]
#                 mask_roi = mask[y0:y1, x0:x1]
                hsv_roi = cv2. cvtColor(roi,cv2. COLOR_BGR2HSV)
                mask_roi = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                #一维直方图
                hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
                #二维直方图
#                 hist = cv2.calcHist( [hsv_roi], [0,2],None, [180,256], [0, 180,0 , 255] )
                
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
                self.hist = hist.reshape(-1)
                #二维直方图显示
#                 plt.imshow(hist,interpolation = 'nearest')
#                 plt.show()
                self.show_hist()

                vis_roi = vis[y0:y1, x0:x1]
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            if self.tracking_state == 1:
                self.selection = None
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
                prob &= mask
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
#                 if track_box[0][1] <= 240:
#             self.ser.write(str(int(track_box[0][0])-320) + " " + str(int(track_box[0][1])-240))
#             print str(int(track_box[0][0])-320) + " " + str(int(track_box[0][1])-240)
                if track_box[1][1] <= 1:
                    self.tracking_state = 0
                    self.start()
                else:
                    if self.show_backproj:
                        vis[:] = prob[...,np.newaxis]
                    try: 
                        cv2.ellipse(vis, track_box, (0, 0, 255), 2)
#                         print track_box
                        a = str(track_box[0][0])+" "+str(track_box[0][1])+" "+str(round(track_box[1][0],2))\
                                       +" "+str(round(track_box[1][1],2))+" "+str(round(track_box[2],2))+"\r\n"
                        print a
#                         self.ser.write(a)
                    except: print track_box

            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
            if ch == ord('r'):
                self.tracking_state = 0
                self.start()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    try: color = sys.argv[1]
    except: color = 1
    print __doc__
    a = App(color)
    a.run()

    
