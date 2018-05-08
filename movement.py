from lib import rovio
import time
import cv2

class Movement(object):

    def __init__(self, rovio,debug=True):
        self.rovio = rovio
        self.debug = debug

        self.movements = []
        self.delay = 1
        self.screen_width = 640
        self.screen_height = 480
        self.screen_height_h = self.screen_height/2
        self.screen_height_q = self.screen_height/4 + self.screen_height_h
        self.screen_width_h = self.screen_width/2

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
    def rotate_left(self,times=1,speed_=4,angle=None):
        if not angle is None:
            times = int(angle/36)
        for i in range(times):
            self.movements.append(self.rovio.rotate_left.__name__)
            self.rovio.rotate_left(speed=speed_)
            time.sleep(self.delay)
    def rotate_left_90(self):
        self.rotate_left(times=2)
        self.rotate_left(speed_=6)
    def rotate_left_less(self):
        self.rotate_left(speed_=7)

    def rotate_right(self, times=1,speed_=4,angle=None):
        print(angle)
        if not angle is None:
            times = int(angle/36)
        print(times)
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
    def move_backward_less(self):
        self.move_more(self.rovio.backward, times=1)

    def move_left(self):
        self.move_more(self.rovio.left)
    def move_left_straight(self):
        self.move_more(self.rovio.left, times=10)
        self.rotate_left_less()

    def move_right(self):
        self.move_more(self.rovio.right)
    def move_right_straight(self):
        self.move_more(self.rovio.right, times=10)
        self.rotate_right_less()

    def stop(self):
        self.rovio.stop()
        print('Caught you!')
        time.sleep(3)

    def move_undo(self, times=1):
        if len(self.movements) > 0:
            movements = self.movements
            for i in range(times):
                movement = self.get_reverse_movement(movements[-1])
                print(movement.__name__, len(movements))
                movement()
                movements = movements[:-1]
            self.movements = movements

    def get_reverse_movement(self, movement):
        return {
            'rotate_left': self.rotate_right,
            'rotate_right': self.rotate_left,
            'forward': self.move_backward,
            'backward': self.move_forward,
            'left': self.move_right,
            'right': self.move_left
        }.get(movement, None)

    def backtrack(self):
        self.move_undo(times=len(self.movements))
