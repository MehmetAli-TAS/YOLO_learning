import cv2
from ultralytics import YOLO

model = YOLO("last.pt") #training file

image = cv2.imread("image.jpg") #image file

results = model(image)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) 
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) 
        label = f"{result.names[int(box.cls)]}: {box.conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("Detected Objects", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
