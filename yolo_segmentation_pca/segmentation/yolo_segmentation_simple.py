from ultralytics import YOLO
import cv2

# Load a model
model = YOLO('yolov8m-seg.pt')  # load an official model

#results = model(source="0",show=True)

results = model('https://ultralytics.com/images/bus.jpg', show=True)  # predict on an image

cv2.waitKey(0)







