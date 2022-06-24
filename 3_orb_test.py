from picamera import PiCamera
import time
import numpy as np
import cv2 as cv
# my modules
import pose_estimation
import robot_control
import utils
import orb
def check_pose(R):
    cond = np.append(R[:,2],R[2,0:2])
    cond[2] -= 1
    cond = abs(cond)
    return not (cond > 0.02).any()
def pixel_click(event,x,y,flags,param): # mouse callback function
    if event == cv.EVENT_LBUTTONDOWN:
        param.compute((x,y))
        print(param.realpt[0], param.realpt[1])
# camera
camera, output = utils.cam_init(480, 640, 1)
# pose estimation
with np.load('param.npz') as data:
    mtx, dist = [data[i] for i in ('mtx','dist')]
with np.load('cam_pose.npz') as data:
    rvec, tvec = [data[i] for i in ('rvec','tvec')]
rmat, _ = cv.Rodrigues(rvec)
transform = pose_estimation.pixel_to_world(mtx, dist, rvec, tvec)
cv.namedWindow('img')
cv.setMouseCallback('img', pixel_click, transform)
# robot control
bot = robot_control.robot_control()
kpt = orb.key_point()
kp_list = []
kp_cap = False
kp_check = False
kp_count = 0
# display
display = utils.vecs_display('display', 4)
for _ in camera.capture_continuous(output, 'yuv', True):
#     output.data = cv.undistort(output.data, mtx, dist, None, None)
    if kp_cap:
        kp_cap = kpt.add_img_loop(output.data)
        if kp_cap == False:
            kp, des = kpt.finish()
            kp_list.append([kp, des])
            kpt.reset()
            print('capture done')
    # find pose
    if kp_check:
        kp2, des2 = kpt.orb.detectAndCompute(output.data,None)
        for item in kp_list:
            matches = kpt.bf.match(item[1],des2)
            print(len(matches)/len(item[0]),len(matches)/len(kp2))
        print('')
        matches = kpt.bf.match(des,des2)
        pts1 = []
        pts2 = []
        for item in matches:
            pts1.append(kp[item.queryIdx].pt)
            pts2.append(kp2[item.trainIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        # from pts2 to pts1, dont know why
        E, _ = cv.findEssentialMat(pts2, pts1, mtx, threshold = 0.2)
        _, R, t, _ = cv.recoverPose(E, pts2, pts1, mtx)
        R = np.matmul(rmat.transpose(),R)
        R = np.matmul(R,rmat)
        t = np.matmul(rmat.transpose(),t)
        ret = check_pose(R)
        if ret:
            kp_check = False
            display.put_vec('r0', R[:,0], 0)
            display.put_vec('r1', R[:,1], 1)
            display.put_vec('r2', R[:,2], 2)
            display.put_vec('t', t, 3)
            display.update()
        else:
            kp_count += 1
            if kp_count >10:
                kp_check = False
                print('check fail')
    cv.imshow('img',output.data)
    key = cv.waitKey(1)
    if key == ord('q'):#quit
        bot.stop()
        break
    if key == ord('c'):#capture keypoint
        kp_cap = True
    elif key == ord('z'):#check keypoint
        kp_count = 0
        kp_check = True
    elif key == ord('r'):#remove keypoint
        kp_list = []
    bot.move_to(transform.realpt) # click control
    bot.move_key(key) # keyboard control
    bot.read() # empty read buffer, prevent freeze up
cv.destroyAllWindows()
camera.close()
