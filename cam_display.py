import numpy as np
import cv2 as cv
import utils
class cam_display:
    def __init__(self, mtx, dist, rvec, tvec):
        self.mtx = mtx
        self.imtx = np.linalg.inv(mtx)
        self.dist = dist
        self.rot, _ = cv.Rodrigues(rvec)
        self.r3 = self.rot[:,2]
        self.rvec = rvec
        self.tvec = tvec
        #self.pixel = np.ones((1,1,2))
        self.realpt = np.zeros((3,1))
        self.prev_pt = (0, 0)
        cv.namedWindow('cam')
        cv.setMouseCallback('cam', self.cam_control)
    def link_bot(self,bot):
        self.bot_pose = bot.pose
        self.target_pose = bot.target_pose
        self.target_change = bot.target_change
    def local_view(self, pixel, criteria_all = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)):
        #self.pixel[0,0,0] = pixel[0]
        #self.pixel[0,0,1] = pixel[1]
        pts = cv.undistortPointsIter(pixel, self.mtx, self.dist, self.mtx, None, criteria_all)
        pts = np.insert(pts,2,1,2)
        pt = pts[0].transpose()
        s = (0 + np.matmul(self.r3,self.tvec)) / np.matmul(self.r3,np.matmul(self.imtx,pt))
        return np.matmul(self.rot.transpose(),(np.matmul(self.imtx,pt)*s-self.tvec))
    def world_view(self, local_pt):
        c, s = np.cos(self.bot_pose.o), np.sin(self.bot_pose.o)
        R = np.array(((c, -s), (s, c)))
        t = np.array((self.bot_pose.x, self.bot_pose.y))
        return np.matmul(R,local_pt[0:2,0]) + t
    def set_target_pose(self, pt):
        local_pt = self.local_view(pt)
        if local_pt[0] > 0 and local_pt[0] < 4000 and abs(local_pt[1]) < 4000:
            self.target_change[0] = True
            self.target_pose.x, self.target_pose.y = self.world_view(local_pt)
            self.target_pose.o = np.arctan2(self.target_pose.y - self.bot_pose.y,self.target_pose.x - self.bot_pose.x)
    def draw_tooltip(self, img):
        local_pt = self.local_view(self.prev_pt)
        if local_pt[0] > 0 and local_pt[0] < 4000 and abs(local_pt[1]) < 4000:
            x, y = self.world_view(local_pt)
            text = str(round(x,2)) + ', ' + str(round(y,2))
            cv.putText(img, text, self.prev_pt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
            cv.circle(img, self.prev_pt, 2, (255,255,255),-1)
    def show(self, display):
        c, s = np.cos(self.bot_pose.o), np.sin(self.bot_pose.o)
        Rt = np.array(((c, s), (-s, c)))
        dt = np.array((self.target_pose.x-self.bot_pose.x,\
                      self.target_pose.y-self.bot_pose.y))
        local_pt = np.matmul(Rt,dt)
        if local_pt[0] > 200 and local_pt[0] < 4000 and abs(local_pt[1]) < 4000:
            self.realpt[0] = local_pt[0]
            self.realpt[1] = local_pt[1]
            self.realpt[2] = 0
            imgpts, _ = cv.projectPoints(self.realpt, self.rvec, self.tvec, self.mtx, self.dist)
            center = np.array((imgpts[0,0,0],imgpts[0,0,1]), np.int32)
            cv.circle(display, center, 10, (0,0,0))
            cv.circle(display, center, 8, (255,255,255))
        self.draw_tooltip(display)
        cv.imshow('cam', display)
    def cam_control(self,event,x,y,flags,param): # mouse callback function
        if event == cv.EVENT_LBUTTONDOWN:
            self.set_target_pose((x, y))
        self.prev_pt = (x, y)
if __name__ == "__main__":
    import robot_control
    # camera
    camera, output = utils.cam_init(480, 640, 1)
    # pose estimation
    with np.load('param.npz') as data:
        mtx, dist = [data[i] for i in ('mtx','dist')]
    with np.load('cam_pose.npz') as data:
        rvec, tvec = [data[i] for i in ('rvec','tvec')]
    
    bot = robot_control.robot_control()
    # display
    display = cam_display(mtx, dist, rvec, tvec)
    display.link_bot(bot)
    for _ in camera.capture_continuous(output, 'yuv', True):
        display.show(output.data)
        key = cv.waitKey(1)
        if key == ord('q'):
            bot.stop()
            break
        if key == ord(' '):
            bot.pause()
            bot.read_pose()
        bot.move_naive()
        bot.move_key(key)
        bot.read_pose()
    cv.destroyAllWindows()
    camera.close()