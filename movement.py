from lib import rovio
import numpy as np
import time
from imgProcessing import process
import cv2
import matplotlib.pyplot as plt

class movementAlgorithm(object):
    def __init__(self, rovio):
        self.rovio = rovio
        self.last = None
        self.key = 0
        self.imgprocess = process.RovioDetect()

    def move_then_rotate(self):
        for i in range(10):
            self.rovio.forward(speed=1)
        time.sleep(0.2)
        for i in range(10):
            self.rovio.rotate_left(speed=4)
            time.sleep(0.5)

    def move_while_rotate(self):
        self.movement(self.rovio.forward_right)
        # self.movement(self.rovio.right)
        self.movement(self.rovio.back_right,times=2)
        self.movement(self.rovio.backward,times=2)
        self.movement(self.rovio.back_left)
        # self.movement(self.rovio.left)
        self.movement(self.rovio.forward_left,times=2)
        self.movement(self.rovio.forward,times=2)

    def rotate_left_36(self,times=1,speed_=4):
        for i in range(times):
            self.rovio.rotate_left(speed=speed_)
            time.sleep(0.5)

    def no_scope(self):
        self.rotate_left_36(10,speed_=4)

    def move_more(self, movement_op):
        for i in range(5): movement_op()
    def movement(self,movement_op,times=1):
        self.rotate_left_36(times)
        self.move_more(movement_op)
        time.sleep(0.3)

    def start(self):
        # self.move_while_rotate()
        # self.move_then_rotate()
        frame = self.rovio.camera.get_frame()
        res = self.imgprocess.isRovio(frame)
        image = cv2.resize(frame,(640,480))
        if(len(res)==0):
            self.rotate_left_36()
        else:
            res = res[0]
            cv2.rectangle(image, (res[2], res[3]), (res[2] + res[4], res[3] + res[5]), (255, 0, 0), 2)
            self.move_more(self.rovio.forward)
        cv2.imshow('Image',image)
        cv2.waitKey(20)
        print(res)
        # time.sleep(2)
