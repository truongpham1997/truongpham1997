import numpy as np
import cv2 as cv
# my modules
import orb
import utils
class fr_data:
    def __init__(self,kp,des,pose):
        self.kp=kp
        self.des=des
        self.pose=pose
class fr_prob:
    def __init__(self,prob,matches,index):
        self.prob=prob
        self.matches=matches
        self.kp_index=index
class prob_bid:
    def __init__(self, limit=8):
        self.reset(limit)
    def add(self,prob,match,index):
        if len(self.list) < self.limit:
            self.list.append(fr_prob(prob,match,index))
        else:
            lindex, lprob = self.lowest_prob_index()
            if prob > lprob:
                self.list[lindex] = fr_prob(prob,match,index)
    def lowest_prob_index(self):
        prob = 1
        index = 0
        for i in range(self.limit):
            if prob > self.list[i].prob:
                prob = self.list[i].prob
                index = i
        return index, prob
    def reset(self, limit=5):
        self.limit = limit
        self.list = []
class key_frame:
    def __init__(self, mtx, rvec, img_size, img_div):
        self.kpt = orb.key_point(img_size, img_div, is_show=True)
        self.mtx = mtx
        self.rmat, _ = cv.Rodrigues(rvec)
        self.kp_list = []
        self.match = prob_bid()
    def global_scan(self, des):
        prob_list = []
        for item in self.kp_list:
            matches = self.kpt.bf.match(item.des,des)
            prob_list.append(len(matches)/len(item.kp))
        return prob_list
    def newframe_scan(self, img, pose, limit=0.8):
        kp, des = self.kpt.detectAndCompute(img)
        cv.drawKeypoints(img, kp, img, flags=1)
        for item in self.kp_list:
            if pose.compare(item.pose): # check location
                return False
            elif abs(utils.rad_in_range(item.pose.o - pose.o)) < 1: # check angle dif < 60*
                matches = self.kpt.bf.match(item.des,des)
                if len(matches)/len(item.kp) > limit:
                    return False
        return True
    def local_scan(self, des, pose, limit=0.3):
        self.match.reset()
        for index in range(len(self.kp_list)):
            if abs(utils.rad_in_range(self.kp_list[index].pose.o - pose.o)) < 1: # check angle dif < 60*
                matches = self.kpt.bf.match(self.kp_list[index].des,des) 
                prob = len(matches)/len(self.kp_list[index].kp)
                if prob > limit:
                    self.match.add(prob,matches,index)
    def check_frame(self, img, pose):
        pending = self.kpt.add_img_loop(img)
        if pending == False:
            kp, des = self.kpt.finish()
            self.local_scan(des, pose)
            angle, diff = self.find_angle(kp, pose.o)
            pose.o = angle # update angle
            print('capture success',angle, diff) # diff low <> precision high, num high <> accuracy high
            self.kp_list.append(fr_data(kp, des, pose.copy()))
            self.kpt.reset()
        return not pending
    def find_angle(self, kp, bot_angle):
        angle_candidate = [bot_angle]
        for item in self.match.list:
            ret, R, _ = self.find_pose(item, kp)
            if ret == False: continue
            angle = utils.rad_in_range(self.kp_list[item.kp_index].pose.o + np.arcsin(R[1,0]))
            if abs(angle) < 10e-5: angle = 0
            angle_candidate.append(angle)
        angle_candidate = sorted(angle_candidate, key = utils.rad_positive)
        if len(angle_candidate) > 3:
            angle = angle_candidate[round(len(angle_candidate)/2)]
        else: angle = bot_angle
        diff = utils.rad_in_range(angle_candidate[len(angle_candidate)-1] - angle_candidate[0])
        return angle, diff
    def find_pose(self,data,kp):# data = fr_prob(prob,matches,kp_index)
        pts1 = []
        pts2 = []
        for item in data.matches:
            pts1.append(self.kp_list[data.kp_index].kp[item.queryIdx].pt)
            pts2.append(kp[item.trainIdx].pt)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        # from pts2 to pts1, dont know why
        E, _ = cv.findEssentialMat(pts2, pts1, self.mtx, threshold = 0.2)
        _, R, t, _ = cv.recoverPose(E, pts2, pts1, self.mtx)
        R = np.matmul(self.rmat.transpose(),R)
        R = np.matmul(R,self.rmat)
        t = np.matmul(self.rmat.transpose(),t)
        ret = self.check_pose(R)
        return ret, R, t
    def check_pose(self,R):
        cond = np.append(R[:,2],R[2,0:2])
        cond[2] -= 1
        cond = abs(cond)
        return not (cond > 0.02).any()
    def get_last(self):
        return self.kp_list[len(self.kp_list)-1]
    def reset(self):
        self.kp_list.clear()
if __name__ == '__main__':
    # my modules
    import pose_estimation
    import robot_control
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
    transform = pose_estimation.pixel_to_world(mtx, dist, rvec, tvec)
    cv.namedWindow('img')
    cv.setMouseCallback('img', pixel_click, transform)
    # robot control
    bot = robot_control.robot_control()
    kfr = key_frame(mtx, rvec)
    kp_cap = False
    kp_check = False
    kp_count = 0
    # display
    display = utils.vecs_display('display', 1)
    for _ in camera.capture_continuous(output, 'yuv', True):
    #     output.data = cv.undistort(output.data, mtx, dist, None, None)
        if kp_cap:
            kp_cap, _= kfr.add_kfr(output.data, bot.pose)
        # find pose
        if kp_check:
            kp, des = kfr.kpt.orb.detectAndCompute(output.data,None)
            prob_list = kfr.global_scan(des)
            print(prob_list)
#             fr = kfr.get_last()
#             ret, R, t = kfr.find_pose(fr, kp, des)
            ret, angle, diff = kfr.find_angle(kp, des, prob_list, 0.1)
            if ret:
                kp_check = False
                display.put_text(ret, 0, 0)
                display.put_text(angle, 0, 1)
                display.put_text(diff, 0, 2)
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
            kfr.reset()
        bot.move_to(transform.realpt) # click control
        bot.move_key(key) # keyboard control
        bot.read() # empty read buffer, prevent freeze up
    cv.destroyAllWindows()
    camera.close()
