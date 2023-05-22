import cv2 as cv
import numpy as np


def detection(image):

    img = image

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_red, upper_red)
    res = cv.bitwise_and(img, img, mask=mask)

    img = cv.medianBlur(res, 5)

    edged = cv.Canny(img, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    return True
