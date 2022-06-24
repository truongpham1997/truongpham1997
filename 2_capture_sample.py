from picamera import PiCamera
import time
import numpy as np
import cv2 as cv
import os
import shutil
# my modules
import pose_estimation
import robot_control
import utils

def pixel_click(event,x,y,flags,param): # mouse callback function
    if event == cv.EVENT_LBUTTONDOWN:
        param.compute((x,y))
        print(param.realpt[0], param.realpt[1])
# camera
camera, output = utils.cam_init(480, 640, 3)
# pose estimation
with np.load('param.npz') as data:
    mtx, dist = [data[i] for i in ('mtx','dist')]
with np.load('cam_pose.npz') as data:
    rvec, tvec = [data[i] for i in ('rvec','tvec')]
transform = pose_estimation.pixel_to_world(mtx, dist, rvec, tvec)
cv.namedWindow('img')
cv.setMouseCallback('img', pixel_click, transform)
# robot control
bot = robot_control.robot_control()
# init capture
path_img, path_vid, vid_i = utils.init_file_system()
img_i = 0
vid_on = False
out = cv.VideoWriter()
# start
for _ in camera.capture_continuous(output, 'bgr', True):
    cv.imshow('img',output.data)
    key = cv.waitKey(1)
    if key == ord('q'):#quit
        bot.stop()
        if camera.recording == True: camera.stop_recording()
        if out.isOpened() == True: out.release()
        break
    if key == ord('r'):#remove
        print('remove data')
        try:shutil.rmtree('data_bk')
        except OSError as e:print(e)
        try:os.rename('data','data_bk')
        except OSError as e:print(e)
        path_img, path_vid, vid_i = utils.init_file_system()
    elif key == ord('c'):#capture
        print('capture %02d'%img_i)
        path_i = 'img_%02d.jpg'%img_i
        cv.imwrite(path_img+path_i,output.data)
        img_i += 1
    elif key == ord('v'):#video
        vid_on = not vid_on
        if vid_on == True:
            print('start recording %02d'%vid_i)
            path_i = 'vid_%02d.h264'%vid_i
            camera.start_recording(path_vid+path_i)
            path_i = 'vid_%02d.avi'%vid_i
            out.open(path_vid+path_i,cv.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))
            vid_i += 1
        else:
            print('stop recording')
            camera.stop_recording()
            out.release()
    if vid_on == True:out.write(output.data)
    bot.move_to(transform.realpt) # click control
    bot.move_key(key) # keyboard control
    bot.read() # empty read buffer, prevent freeze up
cv.destroyAllWindows()
camera.close()
