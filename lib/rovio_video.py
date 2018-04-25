import cv2, time
import urllib.request as urllib
import base64
import numpy as np

class ipCamera(object):

    def __init__(self, url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring(('%s:%s' % (user, password)).encode())[:-1]

        self.req = urllib.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame
