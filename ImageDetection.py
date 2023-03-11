import cv2 as cv


def detection():

    img = cv.imread("Assets\TrafficLights.png")
    cv.imshow("Lights", img)

    cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == "__main__":
    detection()
