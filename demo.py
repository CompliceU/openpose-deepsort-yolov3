#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

############OPENPOSE
# From Python
# It requires OpenCV installed for Python
from sys import platform
import argparse
from filevideostream import FileVideoStream
import time
from fps import FPS
######################################### added cuda
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#########################################

# Import Openpose (Windows/Ubuntu/OSX)
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../build/python/openpose/Release')
        os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../build/x64/Release;' +  dir_path + '/../build/bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "../models/"

# Construct it from system arguments
# op.init_argv(args[1])
# oppython = op.OpenposePython()

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
# Process Image
datum = op.Datum()
imageToProcess = cv2.imread(args[0].image_path)
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop([datum])
####################

def main(yolo):

   # Definition of the parameters
    max_cosine_distance = 0.1 # changed
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    model_filename = 'model_data/mars.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
	
	############
    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture('DJI_0034.mp4')
##################################### added to accelerate
    fps_ = FPS().start()
#####################################

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = 720#int(video_capture.get(3))
        h = 480#int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        frame = cv2.resize(frame, (720, 480))

        cv2.namedWindow("", 0)
        cv2.resizeWindow("", 1980, 1080)

        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs = yolo.detect_image(image)#image)
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker 
		# DeepSort
        tracker.predict()
        tracker.update(detections)
        
		# DeepSort white rectangle
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
		
		# YOLO blue rectangle
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

#############Openpose
        if len(boxs) != 0: # check for yolo
            image_copy_crop = image.copy()
            cropped_img = image_copy_crop.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            cropped_image = np.array(cropped_img)
        else:
            cropped_image = frame
        datum = op.Datum()
        datum.cvInputData = cropped_image
        opWrapper.emplaceAndPop([datum])
        
        if len(boxs) != 0: # check for yolo
            image_copy_paste = Image.fromarray(frame).copy()
            image_copy_paste.paste(Image.fromarray(datum.cvOutputData), (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
            datum.cvOutputData = np.array(image_copy_paste)
            cv2.imshow('', datum.cvOutputData)
            #print("Body keypoints: \n" + str(datum.poseKeypoints))
        else:
            cv2.imshow('', frame)
        fps_.update()
##############
            
        
        if writeVideo_flag:
            # save a frame
            if len(boxs) != 0: # check for yolo
                out.write(datum.cvOutputData)
            else:
                out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    fps_.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
