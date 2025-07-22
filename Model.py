from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8s.pt')

# Train on custom dataset
results = model.train(data="/content/drive/MyDrive/ppe3/data.yaml", epochs=50, imgsz=640)
