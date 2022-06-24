import numpy as np
import cv2 as cv
# my modules
import utils
class map_display:
    def __init__(self,size=600):
        self.background = np.zeros((size,size,3), np.uint8)
        self.map_pose = utils.r_pose(0.0,0.0,0.0)
        self.c , self.s = np.cos(self.map_pose.o), np.sin(self.map_pose.o)
        self.center = size / 2
        self.scale = 1.0
        self.kp_list = []
        cv.namedWindow('map')
        cv.setMouseCallback('map', self.map_control)
        self.mouse_start = True
        self.prev_pt = (0, 0)
    def get_kp_list(self,kp_list):
        self.kp_list = kp_list
    def link_bot(self,bot):
        self.bot_pose = bot.pose
        self.target_pose = bot.target_pose
        self.target_change = bot.target_change
    def show(self):
        map_img = np.copy(self.background)
        self.draw_coord(map_img)
        for item in self.kp_list:
            self.draw_pose(map_img, item.pose, (255,0,0))
        self.draw_pose(map_img, self.target_pose, (0,255,255))
        self.draw_pose(map_img, self.bot_pose, (0,0,255))
        self.draw_tooltip(map_img)
        cv.imshow('map',map_img)
    def draw_tooltip(self, img):
        x, y = self.world_view(self.prev_pt)
        text = str(round(x,2)) + ', ' + str(round(y,2))
        cv.putText(img, text, self.prev_pt, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv.circle(img, self.prev_pt, 2, (255,255,255),-1)
    def follow_pose(self,pose):
        self.map_pose.o = -pose.o
        self.c , self.s = np.cos(self.map_pose.o), np.sin(self.map_pose.o)
        self.map_pose.x = self.s*pose.x + self.c*pose.y
        self.map_pose.y = self.c*pose.x - self.s*pose.y
    def draw_pose(self, img, pose, color):
        pose_t = self.map_view(pose)
        c, s = np.cos(pose_t.o), np.sin(pose_t.o)
        center = np.array((pose_t.x, pose_t.y), np.int32)
        pts = np.int32(center + 100*np.array((c, s))*self.scale)
        radius = np.int32(20*self.scale)
        cv.circle(img, center, radius, color)
        cv.line(img, center, pts, color)
    def map_view(self, pose):
        return utils.r_pose(-pose.o - self.map_pose.o - np.pi/2,\
                            (-self.s*pose.x - self.c*pose.y + self.map_pose.x)*self.scale + self.center,\
                            (-self.c*pose.x + self.s*pose.y + self.map_pose.y)*self.scale + self.center)
    def set_target_pose(self, pt):
        self.target_change[0] = True
        self.target_pose.x, self.target_pose.y = self.world_view(pt)
        self.target_pose.o = np.arctan2(self.target_pose.y - self.bot_pose.y,self.target_pose.x - self.bot_pose.x)
    def world_view(self, pt):
        xt = (pt[0] - self.center)/self.scale - self.map_pose.x
        yt = (pt[1] - self.center)/self.scale - self.map_pose.y
        return -self.s*xt - self.c*yt, -self.c*xt + self.s*yt
    def draw_coord(self, img):
        x_line = self.map_view(utils.r_pose(0.0,0.0,0.0))
        y_line = self.map_view(utils.r_pose(np.pi/2,0.0,0.0))
        
        self.draw_line(img,x_line,(255,255,255))
        self.draw_line(img,y_line,(255,255,255))
    def draw_line(self,img,pose,color):
        c , s = np.cos(pose.o), np.sin(pose.o)
        if c*c < 0.5:
            if s<0:
                y1 = 0
                x1 = int(pose.x - pose.y*c/s)
            else:
                y1= int(2*self.center)
                x1= int(pose.x - (pose.y - y1)*c/s)
        else:
            if c<0:
                x1= int(0)
                y1= int(pose.y - pose.x*s/c)
            else:
                x1= int(2*self.center)
                y1= int(pose.y - (pose.x - x1)*s/c)
#         cv.line(img,(x0,y0),(x1,y1),color)
        cv.line(img,(int(pose.x),int(pose.y)),(x1,y1),color)
    def map_control(self,event,x,y,flags,param): # mouse callback function
        if self.mouse_start:
            self.mouse_start = False
            self.prev_pt = (x, y)
            return
        if event == cv.EVENT_MOUSEWHEEL:#Zoom
            if flags > 0:
                scale = self.scale
                self.scale -= 0.5
                if self.scale < 0.5: self.scale = 0.5
                self.map_pose.x -= (self.scale-scale)*(self.prev_pt[0] - self.center)/(self.scale*scale)
                self.map_pose.y -= (self.scale-scale)*(self.prev_pt[1] - self.center)/(self.scale*scale)
            else:
                scale = self.scale
                self.scale += 0.5
                self.map_pose.x -= (self.scale-scale)*(self.prev_pt[0] - self.center)/(self.scale*scale)
                self.map_pose.y -= (self.scale-scale)*(self.prev_pt[1] - self.center)/(self.scale*scale)
        else:
            if flags & cv.EVENT_FLAG_MBUTTON:
                if flags & cv.EVENT_FLAG_SHIFTKEY:#Rotate
                    o = -(self.center - self.prev_pt[1])*(x - self.prev_pt[0])
                    o += (self.center - self.prev_pt[0])*(y - self.prev_pt[1])
                    o /= (self.center*self.center)
                    self.map_pose.o += o
                    self.c , self.s = np.cos(self.map_pose.o), np.sin(self.map_pose.o)
                    c , s = np.cos(o), np.sin(o)
                    temp = self.map_pose.x
                    self.map_pose.x = c*self.map_pose.x + s*self.map_pose.y
                    self.map_pose.y = -s*temp + c*self.map_pose.y
                else:#Move
                    self.map_pose.x += (x - self.prev_pt[0])/self.scale
                    self.map_pose.y += (y - self.prev_pt[1])/self.scale
            if flags & cv.EVENT_FLAG_LBUTTON:
                self.set_target_pose((x, y))
            self.prev_pt = (x, y)
        if flags & cv.EVENT_FLAG_RBUTTON:
            self.follow_pose(self.bot_pose)
if __name__ == '__main__':
    import pose_estimation
    import key_frame
    import robot_control
    def pixel_click(event,x,y,flags,param): # mouse callback function
        if event == cv.EVENT_LBUTTONDOWN:
            param.compute((x,y))
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
    kfr = key_frame.key_frame(mtx, rvec)
    kp_cap = False
    kp_check = False
    kp_count = 0
    # display
    display = map_display()
    display.link_bot(bot)
    for _ in camera.capture_continuous(output, 'yuv', True):
    #     output.data = cv.undistort(output.data, mtx, dist, None, None)
        if kp_cap:
            kp_cap, ret = kfr.add_kfr(output.data, bot.pose)
            if kp_cap == False:
                if ret:
                    bot.reset_pose()
                    display.get_kp_list(kfr.kp_list)
                bot.pause()
        # find pose
        if kp_check:
            kp_check = False
            kp, des = kfr.kpt.orb.detectAndCompute(output.data,None)
            prob_list = kfr.local_scan(des, bot.pose.o)
#             print(prob_list)
            if not any([prob > 0.8 for prob in prob_list]):
                bot.pause()
                bot.read_pose()
                kp_cap = True
            else:
                print('capture skip')
        display.show()
        cv.imshow('img',output.data)
        key = cv.waitKey(1)
        if key == ord('q'):#quit
            bot.stop()
            break
        elif key == ord('z'):#check keypoint
            kp_check = True
        elif key == ord('r'):#remove keypoint
            kfr.reset()
        bot.move_naive() # click control
        bot.move_key(key) # keyboard control
        bot.read_pose() # empty read buffer, prevent freeze up
    cv.destroyAllWindows()
    camera.close()

