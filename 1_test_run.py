from picamera import PiCamera
import time
import numpy as np
import cv2 as cv
# my modules
import pose_estimation
import robot_control
import utils
def pixel_click(event,x,y,flags,param): # mouse callback function
    if event == cv.EVENT_LBUTTONDOWN:
        param.compute((x,y))
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
# display
display = pose_estimation.vecs_display('display', 1)
for _ in camera.capture_continuous(output, 'bgr', True):
    cv.imshow('img',output.data)
    display.put_vec('realpt', transform.realpt, 0)
    display.update()
    key = cv.waitKey(1)
    if key == ord('q'):#quit
        bot.stop()
        break
    bot.move_to(transform.realpt) # click control
    bot.move_key(key) # keyboard control
    bot.read() # empty read buffer, prevent freeze up
cv.destroyAllWindows()
camera.close()
