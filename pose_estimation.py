# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:01:11 2021

@author: truon
"""

import numpy as np
import cv2 as cv
import glob
import utils

class pattern_pose:
    def __init__(self, mtx, dist, pattern_size, square_size):
        self.mtx = mtx
        self.dist = dist
        self.pattern_size = pattern_size
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        nx = pattern_size[0] #number of inside corners in x
        ny = pattern_size[1] #number of inside corners in y
        self.square_size = square_size
        self.corners = np.zeros((nx*ny,1,2), np.float32)
        self.objp = np.zeros((nx*ny,3), np.float32)
        self.objp[:,:2] = square_size*np.mgrid[0:nx,0:ny].T.reshape(-1,2) # 24mm
        self.rvec = np.zeros((3,1))
        self.tvec = np.zeros((3,1))
    def find(self, img, debug = False, criteria_all = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)):
        # Find the chess board corners
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, self.corners = cv.findChessboardCorners(gray, self.pattern_size, None, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK) # cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS + cv.CALIB_CB_FAST_CHECK
        self.img = img
        if ret == True:
            cv.cornerSubPix(gray, self.corners, (11,11), (-1,-1), criteria_all)
            if debug == True:
                cv.drawChessboardCorners(img, self.pattern_size, self.corners, ret)
            # Find transformation matrix
            ret, self.rvec, self.tvec = cv.solvePnP(self.objp, self.corners, self.mtx, self.dist, flags = cv.SOLVEPNP_ITERATIVE )
            # modify coordinate
            rot, _ = cv.Rodrigues(self.rvec)
            rot[:,[0,1]] = rot[:,[1,0]]
            rot[:,2] = -rot[:,2]
            self.rvec,_ = cv.Rodrigues(rot)
        return ret, self.rvec, self.tvec
    def draw(self, type = 0):
        if type == 0:
            pts_obj = utils.axis_obj(self.square_size)
        else:
            pts_obj = utils.cube_obj(self.square_size)
        imgpts, _ = cv.projectPoints(pts_obj.shape, self.rvec, self.tvec, self.mtx, self.dist)
        self.img = pts_obj.draw(self.img, imgpts)
        return self.img
class pixel_to_world:
    def __init__(self, mtx, dist, rvec, tvec):
        self.mtx = mtx
        self.imtx = np.linalg.inv(mtx)
        self.dist = dist
        self.rot, _ = cv.Rodrigues(rvec)
        self.r3 = self.rot[:,2]
        self.tvec = tvec
        self.pixel = np.ones((1,1,2))
        self.realpt = np.zeros((3,1))
    def compute(self, pixel, criteria_all = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)):
        self.pixel[0,0,0] = pixel[0]
        self.pixel[0,0,1] = pixel[1]
        pts = cv.undistortPointsIter(self.pixel, self.mtx, self.dist, self.mtx, None, criteria_all)
        pts = np.insert(pts,2,1,2)
        pt = pts[0].transpose()
        s = (0 + np.matmul(self.r3,self.tvec)) / np.matmul(self.r3,np.matmul(self.imtx,pt))
        self.realpt = np.matmul(self.rot.transpose(),(np.matmul(self.imtx,pt)*s-self.tvec))
        return self.realpt
    def compute_all(self, pixel, criteria_all = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)):
        pts = cv.undistortPointsIter(pixel, self.mtx, self.dist, self.mtx, None, criteria_all)
        pts = np.insert(pts,2,1,2)
        realpts = np.zeros((pts.shape[0],1,3), np.float32)
        for i in range(pts.shape[0]):
            pt = pts[i].transpose()
            s = (0 + np.matmul(self.r3,self.tvec)) 
            s /= np.matmul(self.r3,np.matmul(self.imtx,pt))
            realpts[i] = np.matmul((np.matmul(self.imtx,pt)*s-self.tvec).transpose(), self.rot)
        return realpts
    def update(self, rvec, tvec):
        self.rot, _ = cv.Rodrigues(rvec)
        self.r3 = self.rot[:,2]
        self.tvec = tvec
def pose_check(pose, transform):
    realpts = transform.compute_all(pose.corners)
    imgpts, _ = cv.projectPoints(realpts, pose.rvec, pose.tvec, pose.mtx, pose.dist)
    error = np.zeros((3,1))
    error[0,0] = cv.norm(imgpts, pose.corners, cv.NORM_L1)/len(imgpts)
    error[1,0] = cv.norm(imgpts, pose.corners, cv.NORM_L2)/len(imgpts)
    error[2,0] = cv.norm(imgpts, pose.corners, cv.NORM_INF)
    return error
if __name__ == "__main__":
    def pixel_click(event,x,y,flags,param): # mouse callback function
        if event == cv.EVENT_LBUTTONDOWN:
            param.compute((x,y))
            print(param.realpt[0], param.realpt[1])
    with np.load('param.npz') as data:
        mtx, dist = [data[i] for i in ('mtx','dist')]
    criteria_all = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    images = glob.glob('calib_check/*.jpg')
    
    img = cv.imread(images[0])
    pose = pattern_pose(mtx, dist, (6,9), 24)
    _, rvec, tvec = pose.find(img)
    pose.draw()
    cv.imshow('img', img)
    transform = pixel_to_world(mtx, dist, rvec, tvec)
    cv.setMouseCallback('img', pixel_click, transform)
    display = utils.vecs_display('display', 4)
    img_no = 0
    while True:
        img = cv.imread(images[img_no])
        _, rvec, tvec = pose.find(img, True)
        transform.update(rvec, tvec)
        error = pose_check(pose, transform)
        pose.draw()
        cv.imshow('img', img)
        display.put_vec('rvec', rvec, 0)
        display.put_vec('tvec', tvec, 1)
        display.put_vec('realpt', (transform.realpt), 2)
        display.put_vec('error', error, 3)
        display.update()
        key = cv.waitKey(1)
        if key == ord('q'):
            break
        elif (key >= ord('0') and key <= ord('9')):
            img_no = key - ord('0')
    cv.destroyAllWindows()
    
