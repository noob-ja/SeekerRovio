# from imgProcessing import process
# import urllib.request
# import numpy as np
# import cv2
#
# obs = process.ObstacleDetect()
#
# # response = urllib.request.urlopen('https://raw.githubusercontent.com/sinameraji/obstacle_detector_opencv/master/pinki.jpeg')
# response = urllib.request.urlopen('https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.C25FhTwkLnW0RG9kyYB32gHaEK%26pid%3D15.1&f=1')
# img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
# frame = cv2.imdecode(img_array, 1)
#
# res = obs(frame)
# print(res)
# cv2.imshow('obs',frame)
# cv2.waitKey(10000)


a = [0,1,2,3]
print(a)
a = a[:-1]
print(a)
