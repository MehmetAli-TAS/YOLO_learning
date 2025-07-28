import cv2
import numpy as np

# Görseli yükle
img = cv2.imread('simple_detection.jpg')

# Görüntüyü griye çevir (siyahı bulmak için)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Eşikleme (siyah bölgeyi ayırmak için)
_, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

# Konturları bul (siyah şekil için)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# En büyük siyah kareyi al
if contours:
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Bilinen kenar uzunluğu (örnek: 2 cm)
    gercek_kenar_cm = 2.0

    # Piksel ölçüsü
    piksel_kenar = w  # kare olduğu için w = h

    # 1 pikselin kaç cm olduğu
    oran = gercek_kenar_cm / piksel_kenar

    # Kırmızı alanı ayırmak için HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Kırmızı piksel sayısı
    kirmizi_piksel = cv2.countNonZero(red_mask)

    # Pikseli cm²'ye çevir
    kirmizi_alan_cm2 = kirmizi_piksel * (oran ** 2)

    print(f"Kırmızı alanın gerçek boyutu: {kirmizi_alan_cm2:.2f} cm²")
else:
    print("Siyah kare bulunamadı.")
