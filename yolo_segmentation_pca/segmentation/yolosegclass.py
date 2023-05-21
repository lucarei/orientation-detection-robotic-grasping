import numpy as np 
from ultralytics import YOLO

class yolo_seg_class:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
    
    def detect(self,img):

        height, width, channels=img.shape

        result=self.model.predict(source=img.copy(),save=False, save_txt=False)
        result=result[0]
        segmentation_contour_idx=[]
        for seg in result.masks.segments:
            seg[:,0] *= width
            seg[:,1] *= height
            segment=np.array(seg, dtype=np.int32)
            segmentation_contour_idx.append(segment)
        
        bboxes=np.array(result.boxes.xyxy.cpu(), dtype="int")

        class_ids=np.array(result.boxes.cls.cpu(),dtype="int")

        scores=np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contour_idx, scores

