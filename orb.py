# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:32:21 2021

@author: truon
"""

import numpy as np
import cv2 as cv
import utils
class key_point:
    bf = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=True)
    
    def __init__(self,img_size,img_div,img_max=10,prob=0.8,is_show=False):
        self.reset(img_max,prob)
        self.is_show = is_show
        self.init_mask(img_size, img_div, 500)
    def add_img_loop(self,img):
        self.add_img(img)
        if self.img_count <= self.img_max: return True
        else: return False
    def add_img_list(self,img_list):
        for img in img_list:
            self.add_img(img)
    def add_img(self,img):
        if self.done: 
            print('class key_point: add_img, already finished')
            return
        self.img_count +=1
        kp, des = self.detectAndCompute(img)
        while len(kp)>0:
            matches = key_point.bf.match(self.des,des)
            # Merge keypoint
            mask = np.ones(len(kp),bool)
            for item in matches:
                mask[item.trainIdx] = False
            kp = [kp[index] for index in range(len(kp)) if mask[index]]
            des = des[mask==True]
            matches = [item for item in matches if item.distance <50]
            if len(matches) == 0:break
            for item in matches:
                self.kp_poll[item.queryIdx] +=1
        if len(kp)>0:
            self.kp += kp
            self.des = np.append(self.des,des,axis=0)
            self.kp_poll = np.append(self.kp_poll, np.zeros(len(kp),np.uint8))
            self.des_new = 0
        else: self.des_new = des.shape[0]
        if(self.is_show):
            kp_show = [self.kp[index] for index in range(len(self.kp)) if self.kp_poll[index] >= round(self.img_count*self.prob)]
            cv.drawKeypoints(img, kp_show, img, flags=1)
            #print('kp_show ', len(kp_show))
    def detectAndCompute(self, img):
        kp = []
        des = np.zeros((len(kp),32),np.uint8)
        for mask in self.mask:
            kp_new, des_new = self.orb.detectAndCompute(img,mask)
            if len(kp_new)>0:
                kp += kp_new
                des = np.append(des,des_new,axis=0)
        return kp, des
    def init_mask(self, img_size, img_div, nfeatures):
        height, width = img_size
        div_row, div_col = img_div
        self.orb = cv.ORB_create(round(nfeatures/(div_col*div_row)))
        h_div = np.ceil(height/div_row).astype(int)
        w_div = np.ceil(width/div_col).astype(int)
        self.mask = [None]*(div_col*div_row)
        for row in range(div_row):
            r1 = row * h_div
            r2 = r1 + h_div
            if r2 > height: r2 = height
            for col in range(div_col):
                c1 = col * w_div
                c2 = c1 + w_div
                if c2 > width: c2 = width
                self.mask[row*div_col + col] = np.zeros(img_size, dtype=np.uint8)
                self.mask[row*div_col + col][r1:r2, c1:c2].fill(255)
    def finish(self):
        if not self.done: 
            self.done = True
            self.kp = [self.kp[index] for index in range(len(self.kp)) if self.kp_poll[index] >= round(self.img_count*self.prob)]
            self.des = self.des[self.kp_poll >= round(self.img_count*self.prob)]
            self.kp_poll = self.kp_poll[self.kp_poll >= round(self.img_count*self.prob)]
        return self.ret()
    def ret(self):
        return self.kp, self.des
    def reset(self,img_max=None,prob=0.8):
        self.kp = []
        self.des = np.zeros((len(self.kp),32),np.uint8) 
        self.kp_poll = np.zeros(len(self.kp),np.uint8) 
        self.img_count = -1;
        if prob > 1: prob = 1
        elif prob < 0: prob = 0
        self.prob = prob
        if img_max is not None:
            self.img_max = img_max
        self.done = False
    def debug(self,img,prob=0.8):
        kp_show = [self.kp[index] for index in range(len(self.kp)) if self.kp_poll[index] >= round(self.img_count*prob)]
        kp_img = cv.drawKeypoints(img, kp_show, None, color=(255,0,0), flags=0)
        print('precision',prob,np.count_nonzero(self.kp_poll >= round(self.img_count*prob))/self.kp_poll.size)
        print('all point',len(self.kp))
        print('confident point',len(kp_show))
        print('new point',self.des_new)
        print('new point/confident point',self.des_new/len(kp_show))
        print('')
        cv.imshow('key_point debug',kp_img)
if __name__ == '__main__':
    with np.load('param.npz') as data:
        mtx, dist = [data[i] for i in ('mtx','dist')]
    camera, output = utils.cam_init(480, 640, 1)
    cv.namedWindow('img')
    kpt = key_point((480,640), (1,2))
    display = utils.vecs_display('display', 1)
    for _ in camera.capture_continuous(output, 'yuv', True):
        kp2, des2 = kpt.detectAndCompute(output.data)
        output.data = cv.drawKeypoints(output.data, kp2, None, flags=0)
        display.put_text(len(kp2), 0, 0)
        display.update()
        cv.imshow('img',output.data)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
    cv.destroyAllWindows()