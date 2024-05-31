import cv2
from google.colab.patches import cv2_imshow
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt') # 학습된 YOLO 모델 불러오기

cap = cv2.VideoCapture('ml_hw.mp4') # 동영상 캡처

# 원본 비디오의 프레임 측정
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('ml_hw_output.mp4', fourcc, 20.0, (frame_width, frame_height))

frame_count = 0
display_interval = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device='cpu') # 객체 감지

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    if frame_count % display_interval == 0:
        cv2_imshow(frame)

    frame_count += 1

cap.release()
out.release()
