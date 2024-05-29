!pip install ultralytics==8.0.196 roboflow

from roboflow import Roboflow

rf = Roboflow(api_key="--------------------") // Roboflow API 키 (숨김처리)

project = rf.workspace("ml-hw-labelling").project("ml-hw-label")
version = project.version(1)

dataset = version.download("yolov8") // YOLOv8 형식의 데이터셋 다운로드

from ultralytics import YOLO

model = YOLO('yolov8n.yaml') // YOLOv8 모델 불러오기

data_path = dataset.location + '/data.yaml' // 데이터 경로 설정

model.train(data=data_path, epochs=100, imgsz=640) // 모델 훈련
