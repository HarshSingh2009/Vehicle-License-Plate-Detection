from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2 as cv


class DetectionPipeline():
    def __init__(self) -> None:
        # Initialize and Load in Custom YOLOv8 model training weights
        self.license_plate_detector = YOLO('YOLOv8_best_weights.pt')  


    def preprocess_image(self, uploaded_file):
        """
        Takes a File in the format - '.jpg' / '.png' / '.jpeg' then converts it into a Numpy array and returns it

        Args:
            uploaded_file: File that needs to be converted to Numpy array

        Returns:
            img_array: Image in the `np.array` format
        """

        img = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(img)

        return img_array
    
    def detectLicensePlates(self, input_array):
        detections = self.license_plate_detector(input_array)[0]
        license_plate_detections = []
        for license_plate in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            license_plate_detections.append([int(x1), int(y1), int(x2), int(y2), score])

        return license_plate_detections
    
    def detections2Image(self, preprocess_image: np.array, detections:list):
        img = np.array(preprocess_image, dtype='uint8')
        for license_plate_info in detections:
            x1, y1, x2, y2, score = license_plate_info
            cv.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        img_detections = np.array(img)
        return img_detections


