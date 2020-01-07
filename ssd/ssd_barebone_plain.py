from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
import cv2

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


# Set the image size.
img_height = 300
img_width = 300


'''
Model loading section
'''

print('Loading the model and settings things up')

K.clear_session()
model = load_model('ssd300_whole.h5', custom_objects={'AnchorBoxes': AnchorBoxes,'DecodeDetections': DecodeDetections, 'L2Normalization': L2Normalization, 'SSDLoss': SSDLoss})

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']


print('I am ready now...')


def recognition(img):

    img = img.reshape(1,img_height,img_width,3)

    y_pred = model.predict(img)

    confidence_threshold = 0.5

    y_pred = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    y_pred = y_pred[0]


    things = []
    conf = []
    coord = []

    for box in y_pred:
        xmin = box[2] * img.shape[2] / img_width
        ymin = box[3] * img.shape[1] / img_height
        xmax = box[4] * img.shape[2] / img_width
        ymax = box[5] * img.shape[1] / img_height
        this_thing = classes[int(box[0])]
        this_conf = box[1]
        ul = [(int(xmin),int(ymin)), (int(xmax),int(ymax))]
        
        things.append(this_thing)
        conf.append(this_conf)
        coord.append(ul)

    return things, conf, coord


'''
Main Program starts from here
'''

cap = cv2.VideoCapture(0)
cv2.namedWindow('Raw')

x_list = []

while(True):
    ret, img = cap.read()
    img = cv2.resize(img,(img_height,img_width))
    
    things, conf, coord = recognition(img)
    
    if((len(things)>0)&(things!=x_list)):
        print(things)
        print(conf)
        x_list= things
    
    for x in coord:
        cv2.rectangle(img, x[0], x[1], (255,0,0), 2)
        
    cv2.imshow('Raw',img)
    
    if cv2.waitKey(1) & 0xFF == ord('o'):
        cap.release()
        cv2.destroyAllWindows()
        break
