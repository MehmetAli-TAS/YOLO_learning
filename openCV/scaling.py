import cv2
from ultralytics import YOLO

uzaklik = 1.0 #girilen resmin gerçek uzaklığı
genislik = 0.40   #girilen resmin gerçek uzaklığı

örnek_resim = "image.jpg" #örnek resim

model = YOLO("last.pt") #model

calib_resim = cv2.imread(örnek_resim)
results = model(calib_resim)
boxes = results[0].boxes.xyxy.cpu().numpy()

x1, y1, x2, y2 = boxes[0]
calib_pixel_width = x2 - x1


odak_uzakligi = (calib_pixel_width * uzaklik) / genislik
print(f"Odak uzaklığı: {odak_uzakligi:.2f}")


görüntü = "img.jpg" #ölçülecek resim
img = cv2.imread(görüntü)
results = model(img)
boxes = results[0].boxes.xyxy.cpu().numpy()



for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    pixel_width = x2 - x1
    distance = (genislik * odak_uzakligi) / pixel_width

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"Mesafe: {distance:.2f} m", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print(f"🎯 Nesne mesafesi: {distance:.2f} m")

# Göster
cv2.imshow("cıktı", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
