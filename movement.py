from lib import Rovio
import numpy as np
import time
import urllib2

class rovioTest(object):
    def __init__(self,url, username, password, port = 80):
        self.rovio = Rovio(url,username=username,password=password,port = port)
        self.last = None
        self.key = 0

    def move_then_rotate(self):
        for i in range(10):
            self.rovio.forward(speed=1)
        time.sleep(0.2)
        for i in range(10):
            self.rovio.rotate_left(speed=4)
            self.downloadImage()
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

    def rotate_left_36(self,times,speed_=4):
        for i in range(times):
            self.rovio.rotate_left(speed=speed_)
            self.downloadImage()
            time.sleep(0.5)

    def no_scope(self):
        self.rotate_left_36(10,speed_=4)

    def move_more(self, movement_op):
        for i in range(5): movement_op()
    def movement(self,movement_op,times=1):
        self.rotate_left_36(times)
        self.move_more(movement_op)
        time.sleep(0.3)

    def main(self):
        start = time.time()
        self.move_while_rotate()
        # self.move_then_rotate()
        end = time.time()
        print(end-start)
        time.sleep(2)

    pictureVal = 1
    ip_of_rovio = '192.168.43.134'
    save_file_path = 'img/'

    def downloadImage(self):
        url = 'http://'+self.ip_of_rovio+'/Jpeg/CamImg0000.jpg'
        request = urllib2.Request(url)
        pic = urllib2.urlopen(request)
        filePath = self.save_file_path + str(self.pictureVal) + '.jpg'
        with open(filePath, 'wb') as localFile:
            localFile.write(pic.read())
        self.pictureVal += 1

if __name__ == "__main__":
    url = '192.168.43.134'
    user = "myname"
    password = "123456"
    app = rovioTest(url, user, password)
    while True:
        app.main()
