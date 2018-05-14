from lib import rovio
import numpy as np
import time
from imgProcessing import process
import cv2
from movement import Movement

class SearchingAlgorithm(object):
    def start(self):
        while True:
            cv2.waitKey(0)
            self.moving()

    def __init__(self, rovio,debug=True):
        self.rovio = rovio
        self.last = None
        self.key = 0
        self.rovioDet = process.RovioDetect()
        self.obsDet = process.ObstacleDetect()
        self.move = Movement(self.rovio)
        self.debug = debug
        self.frame = None

        self.rovio_json = None
        self.rovio_json_ = {'direction':-2}
        self.rovio_found = False
        self.obs_json = None
        self.obs_json_ = {'direction':-2}
        self.obs_found = False

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

    def get_ref(self, json, get_nearest=True):
        # if nothing found
        if isinstance(json, str):
            return json, -2, -2
        else:
            # if getting obstacle refer points
            if isinstance(json, list):
                ref_point = [x['ref_center'][1] for x in json]
                if get_nearest==True:
                    index = ref_point.index(max(ref_point))
                else:
                    index = ref_point.index(min(ref_point))
                ret_ref = json[index]['ref_center']
                direction = json[index]['direction']
                ret_json = json[index]
            else:   #getting rovio refer points
                ret_ref = json['ref_center']
                direction = json['direction']
                ret_json = json
            return ret_ref, direction, ret_json

    def avoidObs(self):
        self.get_frame()
        if self.rovio_found and self.obs_found:
            rovio_ref, rovio_dir, rovio_json = self.get_ref(self.rovio_json)
            obs_ref, obs_dir, obs_json = self.get_ref(self.obs_json)
            if rovio_ref[1] < obs_ref[1]:
                if obs_dir == 1:
                    self.move.move_left_straight()
                else: self.move.move_right_straight()

    def chase(self):
        self.get_frame()
        if self.rovio_json == 'no rovio':
            if self.rovio_json_['direction'] == -1:
                self.move.rotate_left()
            elif self.rovio_json_['direction'] == 1:
                self.move.rotate_right()
            else:
                self.rovio.backward()
        else:
            self.avoidObs()
            self.move_towards('rovio')

    def object_center(self, object, get_nearest=True):
        while True:
            self.get_frame()
            if object == 'rovio':
                if not self.rovio_found:
                    break
                json = self.rovio_json
            else:
                if not self.obs_found:
                    break
                json = self.obs_json
            ret_ref, direction, ret_json = self.get_ref(json, get_nearest=get_nearest)
            obj_center = abs(ret_ref[0] - self.screen_width_h)
            if obj_center > 200:
                if direction==1: self.move.rotate_right(speed_=6)
                else: self.move.rotate_left(speed_=6)
            else:
                break

    def search(self,partial=4,get_nearest=True):
        self.get_frame()
        obs_json = self.obs_json
        if self.rovio_found:
            # init chasing sequence
            self.chase()
            print('')
        else:
            # if not self.obs_found:
            if self.debug: print('no obstacle found')
            obs_json = self.find(get_nearest)
            if obs_json is False:
                if self.debug: print('still no obstacle found')
                # begin traceback operation
                return ''
            elif obs_json is True:
                # found rovio
                return ''
            ref_obs, ref_obs_direction,_ = self.get_ref(obs_json, get_nearest=get_nearest)
            if ref_obs_direction == 1:    # obstacle on the right
                self.move_around(object='obstacle',times=partial, get_nearest=get_nearest)
            elif ref_obs_direction == -1:   # obstacle on the left
                self.move_around(object='obstacle',times=partial, get_nearest=get_nearest)
            else:
                if self.debug: print('facing obstacle')
        self.obs_json_ = obs_json

    def find(self,get_nearest=True):
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
                for obs_ in self.obs_json:
                    obs_['angle'] = angle
                    obs_found.append(obs_)
            self.move.rotate_left()
            angle += 36
            self.get_frame()

        if not (self.rovio_found or len(obs_found) > 0):
            if self.debug: print('rotated 360 degree, cannot find anything')
            return False
        else:
            if self.debug: print('found obs: ',obs_found)
            _,_,obs_to = self.get_ref(obs_found, get_nearest=get_nearest)
            turn_angle = obs_to['angle']
            print(obs_to, turn_angle)
            if turn_angle<=180:
                self.move.rotate_left(angle=turn_angle)
            else:
                self.move.rotate_right(angle=360-turn_angle)
            return obs_to

    def calc_dist(self, json, object='rovio'):
        dist_btm = abs(self.screen_height - json['ref_center'][1])
        if object=='rovio':
            dist_center = abs(self.screen_height_q - json['ref_center_true'][1])
        else:
            dist_center = abs(self.screen_height_h - json['ref_center_true'][1])
        return dist_btm, dist_center

    def move_towards(self, object, get_nearest=True):
        if self.debug:
            print('________________________')
            print('Moving towards: ',object)
        move_stop = 50
        move_near = 100
        while True:
            self.object_center(object=object, get_nearest=get_nearest)
            self.get_frame()
            json = self.rovio_json if object=='rovio' else self.obs_json
            _,_, obj_json = self.get_ref(json,get_nearest=get_nearest)
            if isinstance(obj_json, int):
                if self.debug: print('lost target')
                # self.move.move_undo()
                return False
            obj_dist_btm, obj_dist_center = self.calc_dist(json=obj_json, object=object)
            print(object,'_dist_btm: ', obj_dist_btm)
            print(object,'_dist_center: ', obj_dist_center)
            if obj_dist_btm < move_stop and obj_dist_center < move_stop:
                return True
            elif obj_dist_btm < move_near or obj_dist_center < move_near:
                self.move.move_forward_less()
            else:
                self.move.move_forward()
        if self.debug:
            print('Moving towards done ')
            print('________________________')

    # move around an object: have to move pass it four times to be say move around it
    def move_around(self, object='obstacle',times=4, get_nearest=True):
        if self.debug:
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print('Moving around: ',object)
        if object=='obstacle':
            if not self.move_towards(object=object,get_nearest=get_nearest):
                return ''
            for i in range(times):
                if self.move_pass()==1:
                    break    # move pass rovio
                # attempt to check for rovio
                self.move.move_backward()
                self.get_frame()
                if self.rovio_found: break
                else: self.move.move_forward()

                if i == times-1:
                    break

                if self.moving_direction: self.move.rotate_right_90()
                else: self.move.rotate_left_90()
                self.move.move_backward_less()
            self.moving_direction = not self.moving_direction

        if self.debug:
            print('Moving around done ')
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # move pass an object: have to see it then cant see it to be define as move pass an object
    def move_pass(self, object='obstacle',direction='left'):
        if object=='obstacle':
            found = False
            moved = 0
            while moved < 3:
                ignore = False
                self.get_frame()
                if self.rovio_found: return 1  # found rovio
                obs_found = self.obs_found
                if obs_found:
                    _,_,json = self.get_ref(self.obs_json)
                    obs_dist_btm, obs_dist_center = self.calc_dist(json=json, object=object)
                    print(obs_dist_btm, obs_dist_center)
                    if abs(json['ref_right'][0]-json['ref_left'][0]) < self.screen_width/10:
                        obs_found = False
                    elif obs_dist_center < 50:
                        found = True
                    else:
                        ignore = True
                if found and (not obs_found or ignore):    break
                # move left once, right once
                if self.moving_direction: self.move.move_left_straight()
                else: self.move.move_right_straight()
                moved += 1
            return 0
    '''
    proposed movement algorithm
    move pass an obstacle twice, on DIRECTION ( left or right ), one time left, one time right
    '''
    def moving(self):
        self.moving_direction = True    # True = left, False = right
        get_nearest = True
        while True:
            self.search(partial=3, get_nearest=get_nearest)
            if self.rovio_found:
                print('Aww yeah')
                break
            get_nearest = False
