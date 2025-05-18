from ultralytics import YOLO
import cv2

# Load mô hình đã train (có thể thay bằng 'runs/detect/train/weights/best.pt')
model = YOLO("yolo11n_trained.pt")  # hoặc "yolov11n.pt"

# Mở webcam (0 là default webcam, nếu dùng camera USB có thể là 1 hoặc 2)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán trên khung hình
    results = model.predict(frame, imgsz=640, conf=0.5)  # conf=0.5: ngưỡng confidence

    # Vẽ kết quả lên khung hình
    annotated_frame = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
