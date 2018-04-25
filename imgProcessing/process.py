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

class RovioDetect(object):
    def __init__(self):
        # Set the image size.
        self.img_height = 288
        self.img_width = 352

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

    def isRovio(self, img):
        orig_images = [] # Store the images here.
        input_images = [] # Store resized versions of the images here.

        img = cv2.imread(img)[:, :, ::-1] if isinstance(img, str) else img

        # ONE IMAGE
        orig_images.append(img.copy())
        if img.shape[:2] != (self.img_height, self.img_width):
            img = cv2.resize(img, (self.img_width, self.img_height))
        input_images.append(img)
        input_images = np.array(input_images)

        y_pred = self.model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

        bounding = []
        for bounds in y_pred_thresh[0]:
            xmin = box[2] * orig_images[0].shape[1] / img_width
            ymin = box[3] * orig_images[0].shape[0] / img_height
            xmax = box[4] * orig_images[0].shape[1] / img_width
            ymax = box[5] * orig_images[0].shape[0] / img_height
            bounding.append(xmin,ymin,xmax,ymax,bounds[0])

        # print("Predicted boxes:\n")
        # print('   class   conf xmin   ymin   xmax   ymax')
        return (y_pred_thresh[0])
        # classes = ['background',
        #            'rovio','rovio']
