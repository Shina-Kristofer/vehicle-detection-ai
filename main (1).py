
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# Step 1: Download Required Models from Google Drive
os.system("pip install gdown")

# Google Drive file IDs
classification_id = "12oAd6AF5YOHitewaoB_k5CNdL8BHasjT"
subclass_id = "1x3JpptPMTzCnnxa-mfb_xyUsHK8v4Cto"
yolov8_id = "19kwkHA5tvK5QQ0Fsd3mI6_gf6ED0NLnz"

# Output paths
os.makedirs("models", exist_ok=True)

os.system(f"gdown --id {classification_id} -O models/resnet18_vehicle_cls.pth")
os.system(f"gdown --id {subclass_id} -O models/resnet18_color_subclass.pth")
os.system(f"gdown --id {yolov8_id} -O models/best.pt")

# Set paths to downloaded model files
DET_MODEL_PATH = "/content/drive/MyDrive/yolov8_runs/final_training/weights/best.pt"
TYPE_CLS_MODEL_PATH = "/content/drive/MyDrive/classification_results/resnet18_vehicle_cls.pth"
COLOR_CLS_MODEL_PATH = "/content/drive/MyDrive/classification_results_color/resnet18_color_subclass.pth"

# Vehicle type classes
type_classes = ['Bus', 'Car', 'Motorcycle', 'Truck']

# Color subclass classes
color_subclass_classes = [
    'Bus_black', 'Bus_silver', 'Car_black', 'Car_blue', 'Car_red', 'Car_silver', 'Car_white', 'Car_yellow',
    'Motorcycle_black', 'Motorcycle_red', 'Motorcycle_silver', 'Motorcycle_white', 'Motorcycle_yellow',
    'Truck_black', 'Truck_red', 'Truck_silver', 'Truck_white', 'Extra_class_1', 'Extra_class_2'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load detection model
det_model = YOLO(DET_MODEL_PATH)

# Load vehicle type classification model
type_cls_model = resnet18(pretrained=False)
type_cls_model.fc = nn.Linear(type_cls_model.fc.in_features, len(type_classes))
type_cls_model.load_state_dict(torch.load(TYPE_CLS_MODEL_PATH, map_location=device))
type_cls_model.to(device)
type_cls_model.eval()

# Load color subclass classification model
color_cls_model = resnet18(pretrained=False)
color_cls_model.fc = nn.Linear(color_cls_model.fc.in_features, len(color_subclass_classes))
color_cls_model.load_state_dict(torch.load(COLOR_CLS_MODEL_PATH, map_location=device))
color_cls_model.to(device)
color_cls_model.eval()

# Image transform for classifiers
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = det_model(img)

    boxes = []
    labels = []
    counts = {}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            out_type = type_cls_model(input_tensor)
            _, pred_type = torch.max(out_type, 1)
        vehicle_type = type_classes[pred_type.item()]

        with torch.no_grad():
            out_color = color_cls_model(input_tensor)
            _, pred_color = torch.max(out_color, 1)
        vehicle_color_subclass = color_subclass_classes[pred_color.item()]
        vehicle_color = vehicle_color_subclass.split('_')[1]

        label = f"{vehicle_type} - {vehicle_color}"

        boxes.append([x1, y1, x2, y2])
        labels.append(label)

        counts[vehicle_type] = counts.get(vehicle_type, 0) + 1
        counts[vehicle_color] = counts.get(vehicle_color, 0) + 1

    return JSONResponse(content={"boxes": boxes, "labels": labels, "counts": counts})

