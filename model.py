
import cv2
import numpy as np 
import time
import os
import re
import matplotlib.pyplot as plt

from fastai.vision.all import *
from fastai.metrics import error_rate, accuracy
from pathlib import Path
import pickle

import pathlib

# import the necessary packages
# from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np
import imutils
import cv2

# temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# import pytesseract

modelPath = Path("weights")
# o.path.join(modelPath,'export.pkl')

# state = pickle.load(open(modelPath.joinpath("export.pkl"), 'rb'))

learn=load_learner(modelPath/'export.pkl','rb')
# learn=load_learner(state)


# pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


idx = [
    '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'
    ]


def load_image(image_path,filename):
	original_image = cv2.imread(image_path)
	print("after_loading")
	return original_image





def plate_segmentation(plate):
	V = cv2.split(cv2.cvtColor(plate, cv2.COLOR_BGR2HSV))[2]
	T = threshold_local(V, 29, offset=15, method="gaussian")
	thresh = (V > T).astype("uint8") * 255
	thresh = cv2.bitwise_not(thresh)



	# resize the license plate region to a canonical size
	plate = imutils.resize(plate, width=400)
	thresh = imutils.resize(thresh, width=400)

	# perform a connected components analysis and initialize the mask to store the locations
	# of the character candidates
	labels = measure.label(thresh, connectivity=2, background=0)
	charCandidates = np.zeros(thresh.shape, dtype="uint8")


	for label in np.unique(labels):
  # if this is the background label, ignore it
	  if label == 0:
	    continue

	  # otherwise, construct the label mask to display only connected components for the
	  # current label, then find contours in the label mask
	  labelMask = np.zeros(thresh.shape, dtype="uint8")
	  labelMask[labels == label] = 255
	  cnts,_ = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	  
	  # ensure at least one contour was found in the mask
	  if len(cnts) > 0:
	  # grab the largest contour which corresponds to the component in the mask, then
	  # grab the bounding box for the contour
	    c = max(cnts, key=cv2.contourArea)
	    (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
	    
	    # compute the aspect ratio, solidity, and height ratio for the component
	    # determine if the aspect ratio, solidity, and height of the contour pass
	    # the rules tests

	    height, width = plate.shape[0],plate.shape[1]
	    if height / float(boxH) > 6: continue

	      
	    ratio =  boxH/ float(boxW)
	    #   # if height to width ratio is less than 1.5 skip
	    if ratio < 1.2: continue
	    area = boxH * boxW
	    #   # if width is not more than 25 pixels skip
	    if width / float(boxW) > 18: continue
	    #   # if area is less than 100 pixels skip
	    if area < 100: continue
	    #   # draw the rectangle

	    if boxH < 50 : continue

	    # check to see if the component passes all the tests
	    # if keepAspectRatio and keepSolidity and keepHeight:
	    # compute the convex hull of the contour and draw it on the character
	    # candidates mask
	    hull = cv2.convexHull(c)
	    cv2.drawContours(charCandidates, [hull], -1, 255, -1)

	charCandidates = segmentation.clear_border(charCandidates)

	return charCandidates,thresh

   




def text_getter(plate):
	charCandidates,thresh=plate_segmentation(plate)

	plate_no=""
	cnts,_ = contours, hierarchy = cv2.findContours(charCandidates, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
	  # print(cnts)
	cnts = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
	for cnt in cnts:
	  x,y,w,h = cv2.boundingRect(cnt)
	  roi = thresh[y-5:y+h+5, x-5:x+w+5]

	  roi = cv2.bitwise_not(roi)
	  roi=cv2.resize(roi, (128, 128),interpolation  =cv2.INTER_CUBIC) 

	  cv2.imshow("rect",roi)
	  cv2.waitKey(0)
	  pred=learn.predict(roi)[0]
	  pred=int(pred[-2:])
	   
	  pred=idx[pred-1]
	  plate_no=plate_no+pred


	return plate_no



def plate_dtection(image_path,filename):

	CONFIDENCE_THRESHOLD = 0.2
	NMS_THRESHOLD = 0.4
	COLORS = [(0, 255, 255)]

	class_names = []

	# import unidecode

	classes = []
	net = cv2.dnn.readNet("weights/yolov4-tiny-obj_final.weights", "labels/yolov4-tiny-obj.cfg")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

	model = cv2.dnn_DetectionModel(net)
	model.setInputParams(size=(416, 416), scale=1/255)

	print("Hellp")
	image=load_image(image_path,filename)

	classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

	with open("labels/obj.names", "r") as f:
		# class_name=f.readlines()
			class_names = [line.strip() for line in f.readlines()]

	plate_no=""
	for (classid, score, box) in zip(classes, scores, boxes):
		print("yo yo")

		color = COLORS[int(classid) % len(COLORS)]
		  
		  # print(label)
		xmin, ymin, xmax, ymax = box
		ymin=int(ymin)
		xmax=int(xmax)
		ymin=int(ymin)
		ymax=int(ymax)

		label = "%s : %s"  % (class_names[classid[0]],score)

		cropped_img=image[ymin-5:ymin+ymax+5, xmin-5:xmin+xmax+5]
		 
		   # save image
		plate_no=text_getter(cropped_img)
		  
		print("hello",plate_no)
		  # print(plate_no)
		  ## cv2.imwrite("\\croped.jpg", cropped_img)
		cv2.rectangle(image, (xmin,ymin), (xmin+xmax, ymin+ymax), (0,0,255), 2)

		  
		cv2.putText(image,label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)
		cv2.putText(image,plate_no, (box[0]+10, box[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,120),2 )
	
	output_path = './static/detections/'
	print(output_path)
	print("before_saving img")
	cv2.imwrite(output_path + '{}' .format(filename), image)
	
	return plate_no




d=plate_dtection("E:\\GTA5\\static\\uploads\\00ca4b949dd426f8.jpg",'00ca4b949dd426f8.jpg')
print("plate_no",d)