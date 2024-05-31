!pip install ultralytics==8.0.196 roboflow

from roboflow import Roboflow

# Roboflow API 키 (숨김처리)
rf = Roboflow(api_key="--------------------") 

project = rf.workspace("ml-hw-labelling").project("ml-hw-label")
version = project.version(1)

# YOLOv8 형식의 데이터셋 다운로드
dataset = version.download("yolov8") 

from ultralytics import YOLO

# YOLOv8 모델 불러오기
model = YOLO('yolov8n.yaml') 

# 데이터 경로 설정
data_path = dataset.location + '/data.yaml' 

# 모델 훈련
model.train(data=data_path, epochs=100, imgsz=640)
