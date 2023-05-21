import cv2
import numpy as np

from yolosegclass import yolo_seg_class

img = cv2.imread("/home/lrros2/Desktop/bus.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)



#segmentation detector
ys=yolo_seg_class("yolov8m-seg.pt")

#detects give us four outputs
bboxes, classes, segmentations, scores = ys.detect(img)
print(bboxes)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
        # print("bbox:", bbox, "class id:", class_id, "seg:", seg, "score:", score)
    (x, y, x2, y2) = bbox
    cv2.rectangle(img, (x,y),(x2,y2),(0,0,255),3)
    cv2.polylines(img,[seg],True,(255,0,0),3)


cv2.imshow("image",img)
cv2.waitKey(0)

