import os
import cv2 as cv
import numpy as np


#class to contain some operations to try on frame
class FrameOperations():

    def __init__(self):
        self.CWD = os.getcwd()
        self.RES_F = os.path.join(self.CWD,'resources')
        self.FILTER_F = os.path.join(self.RES_F,'FILTERS')
        self.SPEED_FILTER = cv.imread(os.path.join(self.FILTER_F,"SPEED.png"))
        self.CONT_FILTER = cv.imread(os.path.join(self.FILTER_F,"CONTINUE.png"))


    def average_blur(self,frame,kernel_size):
        conversion = cv.blur(frame,kernel_size)
        return conversion
    
    def gauss_blur(self,frame, kernel_size,sigX):
        conversion = cv.GaussianBlur(frame,kernel_size,sigX)
        return conversion

    def convert_scale_abs(self,frame, alpha, beta):
        """alpha must be float, beta must be int!"""
        #alpha for contrast control, beta for brightness control

        conversion = cv.convertScaleAbs(frame,alpha=alpha,beta=beta)

        return conversion

    def contrast_brightness(self,frame,brightness,contrast):

        conversion = np.int16(frame)
        conversion = conversion * (contrast/127+1) - contrast + brightness
        conversion = np.clip(conversion,0,255)
        # unsigned int
        conversion = np.uint8(conversion)

        return conversion

    def clahe(self,frame):
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv.createCLAHE(clipLimit=3., tileGridSize=(8,8))

        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)  # convert from BGR to LAB color space
        l, a, b = cv.split(lab)  # split on 3 different channels

        l2 = clahe.apply(l)  # apply CLAHE to the L-channel

        lab = cv.merge((l2,a,b))  # merge channels
        conversion = cv.cvtColor(lab, cv.COLOR_LAB2BGR)  # convert from LAB to BGR

        return conversion

    def increase_red(self,frame):
        B,G,R = cv.split(frame)
        B = self.contrast_brightness(B,10,10)
        G = self.contrast_brightness(G,1,1)
        R = self.contrast_brightness(R,1000,1000)


        # merge B,G,R
        higher_red = cv.merge([B,G,R])

        return higher_red

    def apply_filters(self,frame):
        frame_h,frame_w = frame.shape[:2]

        trans_mask = self.CONT_FILTER[:,:,2] == 0
        self.CONT_FILTER[trans_mask] = [-1,-1,-1]

        self.CONT_FILTER = cv.resize(self.CONT_FILTER,(frame_w,frame_h),interpolation=cv.INTER_LINEAR)
        self.SPEED_FILTER = cv.resize(self.SPEED_FILTER,(frame_w,frame_h),interpolation=cv.INTER_LINEAR)

        filtered = cv.addWeighted(frame,1,self.CONT_FILTER,0.3,-15)
        filtered = cv.addWeighted(filtered,0.7,self.SPEED_FILTER,0.3,-15)

        return filtered


    def found_frame_operation(self,frame):
        """Performs all operations on the found frame
        Use if you want to test out multiple options"""

        frame = self.apply_filters(frame)


        return frame

#FO = FrameOperations()
#path = "my_did_it.png"
#img = cv.imread(path)

#img = FO.found_frame_operation(img)

#cv.imshow('in',img)
#cv.waitKey(0)



        
    
    
    
