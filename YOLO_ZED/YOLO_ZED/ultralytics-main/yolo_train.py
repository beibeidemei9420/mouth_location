from ultralytics import YOLO

# load model
model = YOLO('C:/Users/bigfatcat\Desktop/zed-yolo/YOLO_ZED/ultralytics-main/yolov8n.pt')

# Train the model
model.train(data = 'C:/Users/bigfatcat/Desktop/zed-yolo/YOLO_ZED/ultralytics-main/yolo_digger.yaml', workers = 4, epochs = 200, batch = 16)