# -*- coding: utf-8 -*-
"""
Created on Mon May  3 14:47:21 2021

@author: truon
"""
from picamera import PiCamera
import time
import numpy as np
import cv2 as cv
import os
from serial.tools import list_ports

class axis_obj:
    def __init__(self, size):
        self.shape = size*np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,3]])
    def draw(self, img, imgpts):
        imgpts = np.rint(imgpts).astype(int)
        corner = tuple(imgpts[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[3].ravel()), (0,0,255), 5)
        return img
class cube_obj:
    def __init__(self, size):
        self.shape = size*np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                                      [0,0,3], [0,3,3], [3,3,3], [3,0,3] ])
    def draw(self, img, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # draw pillars in blue color
        for i,j in zip(range(4),range(4,8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        return img
class cam_buffer(object):
    def __init__(self,height, width, depth):
        self.h = height
        self.w = width
        self.d = depth
        self.data = np.empty((self.h, self.w, self.d), dtype=np.uint8)
    def write(self, buf):
        self.data = np.frombuffer(
            buf, dtype=np.uint8, count=self.h*self.w*self.d).reshape((self.h, self.w, self.d))
    def flush(self):
        pass
def init_file_system():
    paths = ['data','data/imgs','data/vids']
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)
    vid_i = 0
    while True:
        path = 'data/vids/vid_%02d.h264'%vid_i
        if os.path.isfile(path):vid_i += 1
        path = 'data/vids/vid_%02d.avi'%vid_i
        if os.path.isfile(path):vid_i += 1
        else:break
    img_i = 0
    while True:
        path = 'data/imgs/imgs_%02d'%img_i
        if os.path.isdir(path):
            if len(os.listdir(path)) == 0:break #if path empty
            img_i += 1
        else:
            os.mkdir(path)
            break
    return path+'/', paths[2]+'/', vid_i
def find_port(name, number = 0):
    ports = list(list_ports.comports())
    named_ports_device = list()
    for port in ports:
        if port.manufacturer != None and name in port.manufacturer:
            named_ports_device.append(port.device)
    if len(named_ports_device) == 0:
        return ''
    return named_ports_device[number]
def cam_init(height, width, depth):
    camera = PiCamera(sensor_mode=4, resolution=(width,height))
    time.sleep(2) # let the camera warm up and set gain/white balance
    camera.rotation = 180
    output = cam_buffer(height, width, depth)
    return camera, output
class vecs_display:
    def __init__(self, winname, size):
        self.display = np.zeros((200, size*200), np.uint8)
        self.winname = winname
    def put_vec(self, name, vec, pos):
        value = np.round(vec,2)
        self.put_text(name, pos, 0)
        self.put_text(value[0], pos, 1)
        self.put_text(value[1], pos, 2)
        self.put_text(value[2], pos, 3)
    def put_text(self, text, col, row):
        cv.putText(self.display, str(text), (col*200 + 10,row*50 + 40), cv.FONT_HERSHEY_SIMPLEX, 1, 255)
    def update(self):
        cv.imshow(self.winname, self.display)
        self.display.fill(0)
class r_pose:
    def __init__(self,o,x,y):
        self.o=o
        self.x=x
        self.y=y
    def copy(self):
        return r_pose(self.o,self.x,self.y)
    def compare(self, pose):
        return abs(self.x - pose.x) < 2 and\
               abs(self.y - pose.y) < 2 and\
               abs(rad_in_range(self.o - pose.o)) < 10e-5
def rad_in_range(rad):
    if rad > np.pi:
        rad -= 2*np.pi
        while rad > np.pi:
            rad -= 2*np.pi
    elif rad < -np.pi:
        rad += 2*np.pi
        while rad < -np.pi:
            rad += 2*np.pi
    return rad
def rad_positive(rad):
    if rad < -np.pi:
        rad += 2*np.pi
        while rad < -np.pi:
            rad += 2*np.pi
    return rad
