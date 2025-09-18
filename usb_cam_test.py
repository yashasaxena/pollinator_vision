import cv2

cap = cv2.VideoCapture(0)  # 0 = first camera

if not cap.isOpened():
    print("Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Seawit USB Camera", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
