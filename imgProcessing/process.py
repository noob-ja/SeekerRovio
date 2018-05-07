# Input: image path
# output: list
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import matplotlib.pyplot as plt
from .models.keras_ssd300 import ssd_300
from .keras_loss_function.keras_ssd_loss import SSDLoss
from .keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from .keras_layers.keras_layer_DecodeDetections import DecodeDetections
from .keras_layers.keras_layer_DecodeDetections2 import DecodeDetections2
from .keras_layers.keras_layer_L2Normalization import L2Normalization

from .ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from .data_generator.object_detection_2d_data_generator import DataGenerator
from .data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from .data_generator.object_detection_2d_geometric_ops import Resize
from .data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

import cv2

class RovioDetect(object):
    def __init__(self, screen_height=480, screen_width=640):
        # Set the image size.
        self.img_height = 300
        self.img_width = 300
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.screen_height_h = self.screen_height/2
        self.screen_width_h = self.screen_width/2

        # 1: Build the Keras model

        K.clear_session() # Clear previous models from memory.

        self.model = ssd_300(image_size=(self.img_height, self.img_width, 3),
                        n_classes=2,
                        mode='inference',
                        l2_regularization=0.0005,
                        scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                        aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                                [1.0, 2.0, 0.5],
                                                [1.0, 2.0, 0.5]],
                        two_boxes_for_ar1=True,
                        steps=[8, 16, 32, 64, 100, 300],
                        offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                        clip_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        normalize_coords=True,
                        subtract_mean=[123, 117, 104],
                        swap_channels=[2, 1, 0],
                        confidence_thresh=0.5,
                        iou_threshold=0.45,
                        top_k=200,
                        nms_max_output_size=400)

        # 2: Load the trained weights into the model.

        # TODO: Set the path of the trained weights.
        weights_path = './imgProcessing/rovio_v2.h5'
        self.model.load_weights(weights_path, by_name=True)

        # 3: Compile the model so that Keras won't complain the next time you load it.
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)
        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def detect_rovio(self, img):
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.

        img = cv2.imread(img)[:, :, ::-1] if isinstance(img, str) else img

        colors = plt.cm.hsv(np.linspace(0, 1, 3)).tolist()
        colors = np.array(colors) * 255.0
        colors = colors[:, 1:].astype(np.uint8)

        classes = ['background','rovio','rovio']

        # ONE IMAGE
        orig_images.append(img.copy())
        if img.shape[:2] != (self.img_height, self.img_width):
            img = cv2.resize(img, (self.img_width, self.img_height))
        input_images.append(img)
        input_images = np.array(input_images)

        y_pred = self.model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        # calculations on the bounds of the rovio detected
        boundings = []
        for bounds in y_pred_thresh[0]:
            xmin = int(bounds[2] * orig_images[0].shape[1] / self.img_width)
            ymin = int(bounds[3] * orig_images[0].shape[0] / self.img_height)
            xmax = int(bounds[4] * orig_images[0].shape[1] / self.img_width)
            ymax = int(bounds[5] * orig_images[0].shape[0] / self.img_height)
            color = bounds[0]
            color = colors[int(color)]
            label = '{}: {:.2f}'.format(classes[int(bounds[0])], bounds[1])
            cv2.rectangle(orig_images[0], (xmin, ymin), (xmax, ymax), color.tolist())
            cv2.putText(orig_images[0], label, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, color.tolist())
            boundings.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax, 'class_id': int(bounds[0])})

        return boundings, orig_images[0][:,:,::-1]

    def get_nearest_box(self, img):
        self.boundings, self.processed_frame = self.detect_rovio(img)
        if len(self.boundings)==0:
            return 'no rovio'

        location = []
        # some calculations
        for bounds in self.boundings:
            w = bounds['xmax'] - bounds['xmin']
            h = bounds['ymax'] - bounds['ymin']
            center = {'x': bounds['xmin']+w/2, 'y':bounds['ymin']+h/2}
            bounds['ref_center']= [center['x'],bounds['ymax']]
            bounds['ref_left']  = [bounds['xmin'],bounds['ymax']]
            bounds['ref_right'] = [bounds['xmax'],bounds['ymax']]
            bounds['ref_center_true'] = [center['x'],center['y']]
            bounds['area'] = w * h
            location.append(bounds)

        location_nearest = None
        shortest_distance = 0
        # find the nearest detected rovio
        for bounds in location:
            center_x, center_y = bounds['ref_center_true']
            dist = abs(center_x - self.screen_width_h)
            bounds['dist'] = dist
            if location_nearest is None or dist < shortest_distance:
                shortest_distance = dist
                location_nearest = bounds

        return location_nearest

    def __call__(self,img):
        img = img[:,:,::-1]
        location_nearest = self.get_nearest_box(img)

        # if no rovio found
        if isinstance(location_nearest,str):
            return location_nearest

        # set fuzziness of 1/3 for direction forward
        fuzzy = (location_nearest['xmax']-location_nearest['xmin'])/3
        center_x, center_y = location_nearest['ref_center_true']

        # determining the direction of rovio
        if center_x > self.screen_width_h+fuzzy:    # rovio at right
            location_nearest['direction'] = 1
        elif center_x < self.screen_width_h-fuzzy:  # rovio at left
            location_nearest['direction'] = -1
        else:
            location_nearest['direction'] = 0       # rovio at front

        '''
        location_nearest = [
        {
        'area': Area of rovio,
        'ref_center': Bottom center point,
        'ref_left': Bottom left corner,
        'ref_right': Bottom right corner,
        'ref_center_true': Center of rovio detected
        'direction': Direction of rovio detected
        }
        ]
        '''
        return location_nearest

class ObstacleDetect(object):
    def __init__(self, bound=20, screen_height=480, screen_width=640):
        self.bound = bound
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.screen_height_h = self.screen_height/2
        self.screen_width_h = self.screen_width/2

    def preprocess_image(self,frame):
        # frame = cv2.imread(frame)
        pink = np.uint8([[[147,20,255]]])
        hsv_pink = cv2.cvtColor(pink,cv2.COLOR_BGR2HSV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([hsv_pink[0][0][0]-self.bound,147,147])
        upper_bound = np.array([hsv_pink[0][0][0]+self.bound,255,255])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        res = cv2.bitwise_and(frame,frame, mask=mask)

        # morphology
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        final_result = closing.copy()
        final_result = cv2.cvtColor(final_result, cv2.COLOR_RGB2GRAY)

        return final_result

    def __call__(self,frame):
        processed_frame = self.preprocess_image(frame)
        img, cont, h = cv2.findContours(processed_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        obs = []
        for contour in cont:
            area = cv2.contourArea(contour)
            if (area > 8000):
                # draw the contour
                cv2.drawContours(frame, contour, contourIdx=-1, color=(255, 0, 0), thickness=2, maxLevel=1)
                # draw the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                f = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(f, "Obstacle", (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                # finding the corners
                corners = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
                ref_corners = []
                # display points found
                for corner in corners:
                    rx = corner[0][0]
                    ry = corner[0][1]
                    ref_corners.append(corner[0])
                    cv2.circle(f, (rx, ry), 5, 0, -1)
                    cv2.putText(f, ('(' + str(rx) + ',' + str(ry) + ')'), (rx - 40, ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, True)
                # sort by left and right, and just take bottom points
                ref_corners = sorted(ref_corners, key=(lambda x: [x[1],x[0]]))
                print(ref_corners)
                ref_left = [ref_corners[-2][0], ref_corners[-2][1]]
                ref_right = [ref_corners[-1][0], ref_corners[-1][1]]
                ref_center = [int((ref_left[0] + ref_right[0]) / 2), int((ref_left[1] + ref_right[1]) / 2)]
                ref_center_true = [ref_center[0], int((ref_corners[-2][1]-ref_corners[0][1])/2)]
                # determining the direction of obstacle
                if ref_left[0] > self.screen_width_h:
                    direction = 1
                elif ref_left[0] < self.screen_width_h:
                    direction = -1
                else:
                    direction = 0

                '''
                obs json = [
                {
                'area': Area of obstacle,
                'ref_center': Bottom center point,
                'ref_center_true': True center point of the obstacle
                'ref_left': Bottom left corner,
                'ref_right': Bottom right corner,
                'direction': Direction of obstacle detected
                }
                ]
                '''
                obs.append({'area': area, 'ref_center': ref_center,'ref_center_true': ref_center_true, 'ref_left': ref_left, 'ref_right': ref_right, 'direction': direction})
        if len(obs) != 0:
            return obs
        return 'no obstacle'
