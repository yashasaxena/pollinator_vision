import cv2
import numpy as np

# Load MobileNet SSD
prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# List of class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Set up camera
cap = cv2.VideoCapture(0)  # 0 = first camera

if not cap.isOpened():
    print("Camera not found!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)

            label = f"{CLASSES[idx]}: {confidence*100:.1f}%"
            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            #               (0, 255, 0), 2)
            # cv2.putText(frame, label, (startX, startY - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"{label} detected at ({centerX}, {centerY}) with {confidence} confidence")

    # # Show frame
    # cv2.imshow("MobileNet SSD Detection", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
