#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp
from yolov3.yolov3 import Create_Yolov3
#from yolov3.utils import load_yolo_weights, detect_image, detect_video
from yolov3.configs import *

if __name__ == '__main__': # comment if in UNIX

    image_path   = "./IMAGES/r.3.jpg"
    video_path   = "./IMAGES/test.mp4"

    #yolo = Load_Yolo_model()
    input_size = YOLO_INPUT_SIZE

    yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
    yolo.load_weights("./checkpoints/yolov3_custom_Tiny")


    #detect_image(yolo, image_path, "./IMAGES/r.3_detect.jpg", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    #detect_video(yolo, video_path, './IMAGES/detected.mp4', input_size=YOLO_INPUT_SIZE, show=False, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
    #detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

    #detect_video_realtime_mp(video_path, "Output.mp4", input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)


