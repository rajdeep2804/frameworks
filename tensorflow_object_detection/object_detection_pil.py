import numpy as np
import os
import tensorflow as tf
import cv2
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageDraw, ImageColor


image_path = 'test_card/04500356-dca9-4a86-beb8-1fe5bec0b7e1_817484-Copy1.jpg'
PATH_TO_CKPT = 'saved_model'
NUM_CLASSES = 1
threshold = 0.5

tf.keras.backend.clear_session()

def load_model(PATH_TO_CKPT):
    detect_fn = tf.saved_model.load(PATH_TO_CKPT)
    return detect_fn

def coordinates(width, height, d):
        print('width :' , width)
        print('height :' , height)
        # the box is relative to the image size so we multiply with height and width to get pixels.
        top = d[0] * height
        left = d[1] * width
        bottom = d[2] * height
        right = d[3] * width
        top = int(max(0, np.floor(top + 0.5).astype('int32')))
        left = int(max(0, np.floor(left + 0.5).astype('int32')))
        bottom = int(min(height, np.floor(bottom + 0.5).astype('int32')))
        right = int(min(width, np.floor(right + 0.5).astype('int32')))
        return top, left, bottom, right



def draw_detection(d, c, s, img, height, width):
        """Draw box and label for 1 detection."""
        draw = ImageDraw.Draw(img)
        top, left, bottom, right = coordinates( width,height, d)
        label = label_map[c]
        s = str(round(s, 2))
        label = label+" : "+s
        print('Class : ', label)
        label_size = draw.textsize(label)
        if top - label_size[1] >= 0:
            text_origin = tuple(np.array([left, top - label_size[1]]))
        else:
            text_origin = tuple(np.array([left, top + 1]))
        color = ImageColor.getrgb("green")
        thickness = 3
        draw.rectangle([left + thickness, top + thickness, right - thickness, bottom - thickness], outline=color)
        draw.text(text_origin, label, fill=color)  # , font=font)
        img = np.array(img)
        return img



def img_inference(image_path):
    print(image_path)
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    img_arr = Image.fromarray(img)
    image_dir, image_name = os.path.split(image_path)
    image_np = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input_tensor = np.expand_dims(img, 0)
    detections = detect_fn(input_tensor)
    num_detections = detections['num_detections'].numpy()
    batch_size = num_detections.shape[0]
    bboxes = detections['detection_boxes'][0].numpy()
    bclasses = detections['detection_classes'][0].numpy().astype(np.int32)
    bscores = detections['detection_scores'][0].numpy()
    for detection in range(0, int(num_detections)):
        if bscores[detection] > threshold:
            c = bclasses[detection]
            d = bboxes[detection]
            s = bscores[detection]
            img = draw_detection(d, c, s, img_arr, height, width)
            ress = 'card_output1/'+image_name
            cv2.imwrite(ress, img)
    
label_map = { 1 : 'RC' }   
    
detect_fn = load_model(PATH_TO_CKPT)  

img_inference(image_path)