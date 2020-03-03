import cv2
import numpy

def main():
    img = cv2.imread("data/batch/b00000.tif", cv2.IMREAD_GRAYSCALE)

    thr = cv2.threshold(img, 0,255, cv2.THRESH_OTSU)[1]

    fill = thr.copy()
    cv2.floodFill(fill, None, (0, 0), 255);
    fill = cv2.bitwise_not(fill) | thr

    dist = cv2.distanceTransform(fill, cv2.DIST_L2, 3)

    print(numpy.amax(dist), numpy.amin(dist))

    dist = numpy.amax(dist) - dist

    print(dist.dtype, numpy.amax(dist), numpy.amin(dist))

    while True:
        show(img)
        show(dist)

def show(img):
    cv2.imshow("TEST_contours.py",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
