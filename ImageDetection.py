import cv2 as cv
import numpy as np
import cv2


def detection(image):

    img = image

    lower_red = np.array([160, 100, 100])
    upper_red = np.array([179, 255, 255])

    lower_green = np.array([0, 100, 0])
    upper_green = np.array([50, 255, 50])

    hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yeşil rengi belirlemek için maske oluştur
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Sonuçta yeşil olan alanları göster
    result = cv2.bitwise_and(img, img, mask=green_mask)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red, upper_red)

    pixel_count = cv.countNonZero(mask)
    if pixel_count > 20:
        print("RED")
        # eğer pixel count 20 den büyükse ve bu sırada kırmızı ışık görüntüden çıkıyorsa "ceza kesildi" uyarısı yazsın.
        # Bu sırada kırmızı ışık görüntüden çıkıyorsa "ceza kesildi" uyarısı yazdır
        if pixel_count == 0:
            print("Ceza kesildi!")

        return True
    else:
        #print("green")
        return False
