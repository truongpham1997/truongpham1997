import numpy as np
import cv2 as cv
# my modules
import key_frame
import map_display
import cam_display
import robot_control
import utils

if __name__ == '__main__':
    # camera
    camera, output = utils.cam_init(480, 640, 3)
    # pose estimation
    with np.load('param.npz') as data:
        mtx, dist = [data[i] for i in ('mtx','dist')]
    with np.load('cam_pose.npz') as data:
        rvec, tvec = [data[i] for i in ('rvec','tvec')]
    # robot control
    bot = robot_control.robot_control()
    kfr = key_frame.key_frame(mtx, rvec, (480, 640), (1,2))
    kp_cap = False
    kp_check = False
    kp_count = 0
    # display
    dcam = cam_display.cam_display(mtx, dist, rvec, tvec)
    dmap = map_display.map_display()
    dcam.link_bot(bot)
    dmap.link_bot(bot)
    for _ in camera.capture_continuous(output, 'bgr', True):
    #     output.data = cv.undistort(output.data, mtx, dist, None, None)
        if kp_cap:
            if kfr.check_frame(output.data, bot.pose):
                bot.update_pose()
                dmap.get_kp_list(kfr.kp_list)
                bot.unpause()
                kp_check = True
                kp_cap = False
        elif kp_check: # find pose
            if kfr.newframe_scan(output.data, bot.pose):
                bot.pause()
                kp_check = False
                kp_cap = True
            #else:print('capture skip, after checking 1 image')
        dmap.show()
        dcam.show(output.data)
        key = cv.waitKey(1)
        if key == ord('q'):#quit
            bot.stop()
            break
        elif key == ord('z'):#check keypoint
            kp_check = not kp_check
        elif key == ord('r'):#remove keypoint
            kfr.reset()
            dmap.get_kp_list(kfr.kp_list)
        bot.move_naive() # click control
        bot.move_key(key) # keyboard control
        bot.read_pose() # empty read buffer, prevent freeze up
    cv.destroyAllWindows()
    camera.close()
