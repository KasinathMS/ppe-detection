from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)

model = YOLO(r"C:\Users\user\OneDrive\Documents\Best file\best.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
              'machinery', 'vehicle']
myColor = (0, 0, 255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]
            print(currentClass)

            if conf > 0.5:
                if currentClass in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    myColor = (0, 0, 255)
                elif currentClass in ['Hardhat', 'Mask', 'Person', 'Safety Cone', 'Safety Vest', 'machinery',
                                      'vehicle']:
                    myColor = (0, 255, 0)

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    img = cv2.resize(img, (1920, 1080))
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()