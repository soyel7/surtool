from ultralytics import YOLO

# Load a model
model = YOLO("/home/ydw-3090/change_code/envis/2017/runs/segment/train2/weights/best.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="yolo.yaml", epochs=160, imgsz=1088, degrees = 30,flipud = 0.05, translate = 0.3, shear = 5,batch = 18 )