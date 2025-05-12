import cv2

# Mở webcam (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera!")
    exit()

while True:
    # Đọc từng khung hình từ webcam
    ret, frame = cap.read()  # Thêm dấu ngoặc để gọi phương thức

    if not ret:
        print("Không thể đọc khung hình!")
        break

    # Hiển thị khung hình
    cv2.imshow("Webcam", frame)

    # Dừng lại nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
