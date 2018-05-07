from lib import rovio
import numpy as np
import time
from imgProcessing import process
import cv2

class movementAlgorithm(object):
    def start(self):
        while True:
            self.search()
            time.sleep(5)

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
        self.delay = 1
        self.screen_width = 640
        self.screen_height = 480
        self.screen_height_h = self.screen_height/2
        self.screen_height_q = self.screen_height/4 + self.screen_height_h
        self.screen_width_h = self.screen_width/2

    def show_battery(self, frame):
        sh = frame.shape
        m, n = sh[0], sh[1]
        battery, charging = self.rovio.battery()
        battery = 100 * battery / 130.
        bs = "Battery: %0.1f" % battery
        cs = "Status: Roaming"
        if charging == 80:
            cs = "Status: Charging"
        cv2.putText(frame, bs, (20, 20),  cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
        cv2.putText(frame, cs, (300, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
        return frame

    def move_then_rotate(self):
        for i in range(10):
            self.rovio.forward(speed=1)
        time.sleep(0.2)
        self.rotate_left(times=10)

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
            time.sleep(self.delay)
    def rotate_left_less(self):
        self.rotate_left(speed_=7)

    def rotate_right(self, times=1,speed_=4,angle=None):
        if not angle is None:
            times = int(angle/36)
        for i in range(times):
            self.movements.append(self.rovio.rotate_right.__name__)
            self.rovio.rotate_right(speed=speed_)
            time.sleep(self.delay)
    def rotate_right_90(self):
        self.rotate_right(times=2)
        self.rotate_right(speed_=6)
    def rotate_right_less(self):
        self.rotate_right(speed_=7)

    def move_more(self, movement_op, times=5, speed_=1):
        for i in range(times):
            self.movements.append(movement_op.__name__)
            movement_op(speed=speed_)
        time.sleep(0.3)

    def move_forward(self):
        self.move_more(self.rovio.forward)
    def move_forward_less(self):
        self.move_more(self.rovio.forward, times=1)

    def move_backward(self):
        self.move_more(self.rovio.backward)

    def move_left(self):
        self.move_more(self.rovio.left)
    def move_left_straight(self):
        self.move_more(self.rovio.left, times=10)
        self.rotate_left_less()

    def move_right(self):
        self.move_more(self.rovio.right)

    def stop(self):
        self.rovio.stop()
        print('Caught you!')
        time.sleep(3)

    def move_undo(self):
        if len(self.movements) > 0:
            movement = self.get_reverse_movement(self.movements[-1])
            movement()

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
        if not self.rovio_json is None: self.rovio_json_ = self.rovio_json
        if not self.obs_json is None: self.obs_json_ = self.obs_json
        self.rovio_json = self.rovioDet(self.frame)
        self.rovio_found = not isinstance(self.rovio_json, str)
        self.obs_json = self.obsDet(self.frame)
        self.obs_found = not isinstance(self.obs_json, str)
        if not self.frame is None:
            self.frame = self.show_battery(self.frame)
            cv2.imshow('Obstacle', self.frame)
        if not self.rovioDet.processed_frame is None:
            # self.rovioDet.processed_frame = self.show_battery(self.rovioDet.processed_frame)
            cv2.imshow('Rovio', self.rovioDet.processed_frame)
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
            self.object_center()
            self.move_towards('rovio')

    def object_center(self, object='rovio'):
        if object=='rovio':
            while True:
                self.get_frame()
                ret_ref, direction, ret_json = self.get_ref(self.rovio_json)
                rovio_center = abs(ret_ref[0] - self.screen_width_h)
                if rovio_center > 150:
                    if direction==1: self.rotate_right(speed_=6)
                    else: self.rotate_left(speed_=6)
                else:
                    break

    def get_ref(self, json, get_nearest=True):
        # if nothing found
        if isinstance(json, str):
            return json, -2, -2
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
                ret_json = json[index]
            else:   #getting rovio refer points
                ret_ref = json['ref_center']
                direction = json['direction']
                ret_json = json
            return ret_ref, direction, ret_json

    def search(self):
        self.get_frame()
        obs_json = self.obs_json
        if self.rovio_found:
            # init chasing sequence
            self.chase()
            print('')
        else:
            if not self.obs_found:
                if self.debug: print('no obstacle found')
                obs_json = self.find()
                if obs_json is False:
                    if self.debug: print('still no obstacle found')
                    # begin traceback operation
                    return ''
                elif obs_json is True:
                    # found rovio
                    return ''
            ref_obs, ref_obs_direction,_ = self.get_ref(obs_json, get_nearest=True)
            if ref_obs_direction == 1:    # obstacle on the right
                self.move_around(object='obstacle')
            elif ref_obs_direction == -1:   # obstacle on the left
                self.move_around(object='obstacle')
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
                return True
            if self.obs_found:
                if self.debug: print('found obstacle at angle ', angle)
                self.obs_json[0]['angle'] = angle
                obs_found += self.obs_json
            self.rotate_left()
            angle += 36
            self.get_frame()

        if not (self.rovio_found or len(obs_found) > 0):
            if self.debug: print('rotated 360 degree, cannot find anything')
            return False
        else:
            if self.debug: print('found obs: ',obs_found)
            nearest_obs = {'area':-1, 'angle':0}
            for obs_ in obs_found:
                if obs_['area'] > nearest_obs['area']:
                    nearest_obs = obs_
            turn_angle = nearest_obs['angle']
            if turn_angle<=180:
                self.rotate_left(angle=turn_angle)
            else:
                self.rotate_right(angle=turn_angle-180)
            return nearest_obs

    def move_towards(self, object):
        move_stop = 50
        move_near = 100
        while True:
            self.get_frame()
            json = self.rovio_json if object=='rovio' else self.obs_json
            obj_ref, obj_dir, obj_json = self.get_ref(json)
            if isinstance(obj_json, int):
                if self.debug: print('lost target')
                self.move_undo()
                break
            obj_dist_btm = abs(self.screen_height - obj_ref[1])
            if object=='rovio':
                obj_dist_center = abs(self.screen_height_q - obj_json['ref_center_true'][1])
            else:
                obj_dist_center = abs(self.screen_height_h - obj_json['ref_center_true'][1])
            print(object,'_dist_btm: ', obj_dist_btm)
            print(object,'_dist_center: ', obj_dist_center)
            if obj_dist_btm < move_stop and obj_dist_center < move_stop:
                break
            elif obj_dist_btm < move_near or obj_dist_center < move_near:
                self.move_forward_less()
            else:
                self.move_forward()

    # move around an object: have to move pass it four times to be say move around it
    def move_around(self, object='obstacle'):
        if object=='obstacle':
            self.move_towards(object=object)
            for i in range(4):
                self.move_pass()    # move pass rovio
                # attempt to check for rovio
                self.move_backward()
                self.get_frame()
                if self.rovio_found: break
                else: self.move_forward()

                self.rotate_right_90()

    # move pass an object: have to see it then cant see it to be define as move pass an object
    def move_pass(self, object='obstacle',direction='left'):
        if object=='obstacle':
            found = False
            while True:
                self.get_frame()
                if self.rovio_found: break  # found rovio
                if found and not self.obs_found:    break
                if self.obs_found:  found = True
                self.move_left_straight()
