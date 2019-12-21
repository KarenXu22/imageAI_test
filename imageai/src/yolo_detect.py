from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "pretrained-yolov3.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "5.jpg"), output_image_path=os.path.join(execution_path , "5_detect.jpg"))

for eachObject in detections:
    print(eachObject["name"], " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"])
