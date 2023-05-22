import cv2 as cv
import numpy as np


def detection(image):

    img = image

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red, upper_red)

    pixel_count = cv.countNonZero(mask)
    if pixel_count > 20:
        return True
    else:
        return False
