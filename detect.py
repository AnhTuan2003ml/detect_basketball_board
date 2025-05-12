import torch
import cv2

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
    
    # Vẽ kết quả lên frame
    results.render()  # Dự đoán và vẽ bounding box lên ảnh
    
    # Hiển thị ảnh với các bounding box
    cv2.imshow('YOLOv5 Detection', results.ims[0])  # Sử dụng results.ims[0] thay vì results.imgs[0]
    
    # Dừng khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ khi thoát
cap.release()
cv2.destroyAllWindows()
