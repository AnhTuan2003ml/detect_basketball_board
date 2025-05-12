import torch
import cv2
import numpy as np

# Tải mô hình YOLOv5 đã được huấn luyện
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt')

# Khởi tạo camera (0 là ID của webcam, bạn có thể thay đổi nếu sử dụng camera khác)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Tiến hành phát hiện vật thể với ảnh hiện tại từ webcam
    results = model(frame)
    
    # Lấy thông tin bounding box (x_center, y_center, width, height, confidence)
    boxes = results.xywh[0]  # (x_center, y_center, width, height, confidence, class)
    
    # In ra cấu trúc của boxes để kiểm tra
    print(boxes)

    # Nếu phát hiện đối tượng có confidence > 0.7
    for box in boxes:
        # Kiểm tra số lượng phần tử trong box
        if len(box) >= 5:
            x_center, y_center, width, height, confidence = box[:5]  # Lấy 5 phần tử đầu tiên

            if confidence > 0.6:
                # Tính tọa độ góc trên bên trái của bounding box
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Vẽ bounding box trên ảnh
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Hiển thị thông tin
                print(f"Bounding Box Center: (x: {x_center}, y: {y_center})")
                print(f"Confidence: {confidence}")
    
    # Hiển thị ảnh với các bounding box đã được vẽ
    cv2.imshow('YOLOv5 Detection', frame)
    
    # Dừng khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ khi thoát
cap.release()
cv2.destroyAllWindows()
