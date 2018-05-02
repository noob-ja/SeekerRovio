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

        self.rovio_json = None
        self.rovio_json_ = {'direction':-2}
        self.rovio_found = False
        self.obs_json = None
        self.obs_json_ = {'direction':-2}
        self.obs_found = False

        self.movements = []

    def move_then_rotate(self):
        for i in range(10):
            self.rovio.forward(speed=1)
        time.sleep(0.2)
        for i in range(10):
            self.rovio.rotate_left(speed=4)
            time.sleep(0.5)

    # def move_while_rotate(self):
    #     self.movement(self.rovio.forward_right)
    #     # self.movement(self.rovio.right)
    #     self.movement(self.rovio.back_right,times=2)
    #     self.movement(self.rovio.backward,times=2)
    #     self.movement(self.rovio.back_left)
    #     # self.movement(self.rovio.left)
    #     self.movement(self.rovio.forward_left,times=2)
    #     self.movement(self.rovio.forward,times=2)

    '''
    movements
    '''
    # def movement(self,movement_op,times=1):
    #     self.rotate_left_36(times)
    #     self.move_more(movement_op)
    #     time.sleep(0.3)
    def rotate_left(self,times=1,speed_=4,angle=None):
        if not angle is None:
            times = int(angle/36)
        for i in range(times):
            self.movements.append(self.rovio.rotate_left.__name__)
            self.rovio.rotate_left(speed=speed_)
            time.sleep(0.5)

    def rotate_right(self, times=1,speed_=4,angle=None):
        for i in range(times):
            self.movements.append(self.rovio.rotate_right.__name__)
            self.rovio.rotate_right(speed=speed_)
            time.sleep(0.5)

    def move_more(self, movement_op, times=5, speed_=1):
        self.movements.append(movement_op.__name__)
        for i in range(times): movement_op(speed=speed_)
        time.sleep(0.3)

    def move_forward(self):
        self.move_more(self.rovio.forward)

    def move_backward(self):
        self.move_more(self.rovio.backward)

    def move_left(self):
        self.move_more(self.rovio.left)

    def move_right(self):
        self.move_more(self.rovio.right)

    def stop(self):
        self.rovio.stop()
        print('Caught you!')
        time.sleep(3)

    def start(self):
        while True:
            # self.chase()
            self.search()
            print(self.movements)
            if len(self.movements)>0: print(self.get_reverse_movement(self.movements[-1]))

    def get_reverse_movement(self, movement):
        return {
            'rotate_left': self.rotate_right,
            'rotate_right': self.rotate_left,
            'forward': self.move_backward,
            'backward': self.move_forward,
            'left': self.move_right,
            'right': self.move_left
        }.get(movement, None)

    def get_frame(self):
        if self.debug:
            print('=============================================')
        self.frame = None
        while self.frame is None:
            self.frame = self.rovio.camera.get_frame()
        self.rovio_json = self.rovioDet(self.frame)
        self.rovio_found = not isinstance(self.rovio_json, str)
        self.obs_json = self.obsDet(self.frame)
        self.obs_found = not isinstance(self.obs_json, str)
        if self.frame is not None:
            cv2.imshow('Live', self.frame)
        # if self.rovioDet.processed_frame is not None:
        #     cv2.imshow('Rovio', self.rovioDet.processed_frame)
        cv2.waitKey(1)
        if self.debug:
            print('rovio: ', self.rovio_json)
            print('obs: ', self.obs_json)
            print('=============================================')

    def chase(self):
        self.get_frame()
        if self.rovio_json == 'no rovio':
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

    def get_ref(self, json, get_nearest=True):
        # if nothing found
        if isinstance(json, str):
            return json, -2
        else:
            # if getting obstacle refer points
            if isinstance(json, list):
                areas = [x['area'] for x in json]
                if get_nearest==True:
                    index = areas.index(max(areas))
                else:
                    index = areas.index(min(areas))
                ret_ref = json[index]['ref_center']
                direction = json[index]['direction']
            else:   #getting rovio refer points
                ret_ref = json['ref_center']
                direction = json['direction']
            return ret_ref, direction

    def search(self):
        self.get_frame()
        if self.rovio_found:
            # init chasing sequence
            print('')
        else:
            if not self.obs_found:
                if self.debug: print('no obstacle found')
                obs_json = self.find()
                if obs_json is None:
                    if self.debug: print('still no obstacle found')
                    return ''
                else:
                    obs_json = self.obs_json
            obs_json = self.obs_json
            ref_obs, ref_obs_direction = self.get_ref(obs_json, get_nearest=True)
            if ref_obs_direction == 1:    # obstacle on the right
                self.move_forward()
            elif ref_obs_direction == -1:   # obstacle on the left
                self.move_forward()
            else:
                if self.debug: print('facing obstacle')
        self.obs_json_ = obs_json

    def find(self):
        self.get_frame()
        angle = 0
        obs_found = []
        # rotate and look for obs or rovio
        while angle<360:
            if self.rovio_found:
                if self.debug: print('found rovio at angle ', angle)
                break
            if self.obs_found:
                if self.debug: print('found obstacle at angle ', angle)
                self.obs_json[0]['angle'] = angle
                obs_found += self.obs_json
            self.rotate_left()
            angle += 36
            self.get_frame()

        if not (self.rovio_found or len(obs_found) > 0):
            if self.debug: print('rotated 360 degree, cannot find anything')
            return None
        else:
            if self.debug: print('found obs: ',obs_found)
            nearest_obs = {'area':-1, 'angle':0}
            for obs_ in obs_found:
                if obs_['area'] > nearest_obs['area']:
                    nearest_obs = obs_
            turn_angle = nearest_obs['angle']
            if angle<=180:
                self.rotate_left(angle=angle)
            else:
                self.rotate_right(angle=angle-180)
            return nearest_obs
