import cv2, time
import urllib.request as urllib
import base64
import numpy as np
import threading

class ipCamera(object):

    def __init__(self, url, user = None, password = None, debug=False):
        self.url = url
        self.auth_encoded = base64.encodestring(('%s:%s' % (user, password)).encode())[:-1]
        self.debug = debug
        self.curr_frame = None
        self.thread = threading.Thread(target=self.stream_frame)
        self.thread.setDaemon(True)
        self.thread.start()

    def stream_frame(self):
        request = urllib.Request(self.url)
        request.add_header('Authorization', 'Basic %s' % self.auth_encoded)
        while True:
            try:
                response = urllib.urlopen(request)
                img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
                frame = cv2.imdecode(img_array, 1)
                self.curr_frame = frame
                if self.debug:
                    cv2.imshow('Raw frame',self.curr_frame)
                    cv2.waitKey(1)
            except Exception as e:
                print('Error')

    def get_frame(self,img_width=640, img_height=480):
        frame = self.curr_frame.copy() if self.curr_frame is not None else None
        frame = cv2.resize(frame, (img_width, img_height))
        return frame
