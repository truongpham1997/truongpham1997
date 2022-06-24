import numpy as np
import serial
import cv2 as cv
import re
# my modules
import utils
class robot_control:
    def __init__(self, port_order=0):
        port = utils.find_port('Arduino',port_order)
        if len(port) == 0:
            print('No Arduino found')
            self.is_connected = False
        else:
            self.pose = utils.r_pose(0.0,0.0,0.0)
            self.target_pose = utils.r_pose(0.0,0.0,0.0)
            self.target_change = [False]
            self.is_connected = True
            self.ser = serial.Serial(port, 38400, timeout=1)
            self.ser.flush()
            self.update_pose()
            self.last_key = ord('s')
            self.serial_dict = { ord('a'): { ord('w'):'x200o50', ord('x'):'x-200o-50', ord('s'):'x0o100' },
                             ord('d'): { ord('w'):'x200o-50', ord('x'):'x-200o50', ord('s'):'x0o-100' }}
    def move_naive(self):
        if self.target_change[0]:
            self.target_change[0] = False
            comm = 'm1'
            comm += 'x'+str(np.rint(self.target_pose.x).astype(int))
            comm += 'y'+str(np.rint(self.target_pose.y).astype(int))
            comm += '\n'
            self.ser.write(comm.encode('utf-8'))
    def rotate_naive(self, rot):
        self.target_pose.o = self.pose.o + rot*np.pi/180
        comm = 'm1'
        comm += 'o'+str(np.rint(self.target_pose.o*180/np.pi).astype(int))
        comm += '\n'
        self.ser.write(comm.encode('utf-8'))
    def move_key(self,key):
        if self.is_connected == True:
            comm = self._get(key)
            if comm != '':
                comm = 'm0' + comm + '\n'
                self.ser.write(comm.encode('utf-8'))
    def _get(self, key):
        command = ''
        if key == ord('s'):
            command = 'x0o0'
            self.last_key = key
        elif key == ord('w'):
            command = 'x200o0'
            self.last_key = key
        elif key == ord('x'):
            command = 'x-200o0'
            self.last_key = key
        elif key == ord('a') or key == ord('d'):
            command = self.serial_dict[key][self.last_key]
        return command
    def pause(self):
        if self.is_connected == True:
            self.ser.write("s1\n".encode('utf-8'))
            self.ser.flush()
    def unpause(self):
        if self.is_connected == True:
            self.ser.write("s2\n".encode('utf-8'))
            self.ser.flush()
    def stop(self):
        if self.is_connected == True:
            self.ser.write("m2x0y0o0m1\n".encode('utf-8'))
            self.ser.flush()
            self.ser.close()
    def read(self):
        line = ''
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8','ignore').rstrip()
        return line
    def read_pose(self):
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8','ignore').rstrip()
            value_list = re.findall(r"[-+]?\d*\.?\d+", line)
            if len(value_list) == 3:
                self.pose.x = float(value_list[0])
                self.pose.y = float(value_list[1])
                self.pose.o = float(value_list[2]) * np.pi / 180
    def update_pose(self):
        comm = 'm2'
        comm += 'x'+str(round(self.pose.x,4))
        comm += 'y'+str(round(self.pose.y,4))
        comm += 'o'+str(round(self.pose.o * 180 / np.pi,4))
        comm += '\n'
        self.ser.write(comm.encode('utf-8'))
        self.target_change[0] = True
        self.ser.flush()
        pose = utils.r_pose(0.0,0.0,0.0)
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8','ignore').rstrip()
                value_list = re.findall(r"[-+]?\d*\.?\d+", line)
                if len(value_list) == 3:
                    pose.x = float(value_list[0])
                    pose.y = float(value_list[1])
                    pose.o = float(value_list[2]) * np.pi / 180
                    if pose.compare(self.pose): break
if __name__ == "__main__":
    # camera
    camera, output = utils.cam_init(480, 640, 1)
    # pose estimation
    with np.load('param.npz') as data:
        mtx, dist = [data[i] for i in ('mtx','dist')]
    with np.load('cam_pose.npz') as data:
        rvec, tvec = [data[i] for i in ('rvec','tvec')]
    cv.namedWindow('img')
    bot = robot_control()
    for _ in camera.capture_continuous(output, 'yuv', True):
        cv.imshow('img',output.data)
        key = cv.waitKey(1)
        if key == ord('q'):
            bot.stop()
            break
        if key == ord(' '):
            bot.pause()
            bot.read_pose()
        bot.move_key(key)
        bot.read_pose()
    cv.destroyAllWindows()
    camera.close()