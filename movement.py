from lib import rovio
import numpy as np
import time
from imgProcessing import process
import cv2

class movementAlgorithm(object):
    def __init__(self, rovio,debug=True):
        self.rovio = rovio
        self.last = None
        self.key = 0
        self.rovioDet = process.RovioDetect()
        self.obsDet = process.ObstacleDetect()
        self.debug = debug
        self.frame = None

        self.rovio_json_ = {'direction':-1}

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

    def rotate_left(self,times=1,speed_=4):
        for i in range(times):
            self.rovio.rotate_left(speed=speed_)
            time.sleep(0.5)
    def rotate_right(self, times=1,speed_=4):
        for i in range(times):
            self.rovio.rotate_right(speed=speed_)
            time.sleep(0.5)

    def move_more(self, movement_op, times=5):
        for i in range(times): movement_op(speed=1)
    def movement(self,movement_op,times=1):
        self.rotate_left_36(times)
        self.move_more(movement_op)
        time.sleep(0.3)

    def stop(self):
        self.rovio.stop()
        print('Caught you!')
        time.sleep(3)

    def start(self):
        while True:
            self.chase()

    def get_frame(self):
        if self.debug:
            print('=============================================')
        self.frame = None
        while self.frame is None:
            self.frame = self.rovio.camera.get_frame()
        self.rovio_json = self.rovioDet(self.frame)
        # self.obstacle_direction = self.obsDet(self.frame)
        # if self.frame is not None:
            # cv2.namedWindow('Obstacle')
        #     cv2.imshow('Obstacle', self.frame)
        if self.rovioDet.processed_frame is not None:
            cv2.imshow('Rovio', self.rovioDet.processed_frame)
        cv2.waitKey(1)
        if self.debug:
            print('rovio: ', self.rovio_json)
            # print('obs: ',self.obstacle_direction)
            print('=============================================')

    def chase(self):
        self.get_frame()
        if self.rovio_json=='no rovio':
            if self.rovio_json_['direction'] == -1:
                self.rotate_left()
            elif self.rovio_json_['direction'] == 1:
                self.rotate_right()
            else:
                self.rovio.backward()
        else:
            rovio_size = self.rovio_json['xmax']-self.rovio_json['xmin']
            self.rovio_json_ = self.rovio_json
            print('rovio_size: ', rovio_size)
            if rovio_size > 500:
                self.stop()
            elif rovio_size > 400:
                self.move_more(self.rovio.forward, times=3)
            elif rovio_size < 250:
                self.move_more(self.rovio.forward, times=10)
            else:
                self.move_more(self.rovio.forward)
