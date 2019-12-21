from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
#detector.setModelPath("hololens-ex-60--loss-2.76.h5")
detector.setModelPath("detection_model-ex-10--loss-3.89.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="5.jpg", output_image_path="5_holo.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
