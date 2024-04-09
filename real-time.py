from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import cv2
model=YOLO("C:/Users/msi/Desktop/projet semesterielle/best.pt")
model.predict(source="0",show=True,conf=0.5)