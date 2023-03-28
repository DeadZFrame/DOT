import cv2 as cv
import numpy as np


def detection():

    img = cv.imread("Assets\TrafficLights.png")

    lower_yellow = np.array([0, 50, 50])
    upper_yellow = np.array([10, 255, 255])

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    res = cv.bitwise_and(img, img, mask=mask)

    img = cv.medianBlur(res, 5)

    edged = cv.Canny(img, 30, 200)
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    x, y = int(1200 / 1.5), int(1172 / 1.5)

    cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL)
    cv.resizeWindow("Resized_Window", x, y)
    cv.imshow("Resized_Window", img)

    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    detection()
