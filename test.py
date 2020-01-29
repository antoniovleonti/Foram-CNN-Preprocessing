# Antonio Leonti & Grayson Kelly
# 1.29.2020
#

import numpy
import cv2
from matplotlib import pyplot as plt

def main():
    #import image
    img = cv2.imread("data/train/amp_radiarta/amp_radiarta_100496.png", cv2.IMREAD_GRAYSCALE)
    #threshold
    thr = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(thr, cv2.DIST_L2, 3)
    #sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)[1]

    show_image("thr", thr)
    show_image("dist", dist_transform)
    #show_image("sure_fg", sure_fg)

def show_image(str, img):
    cv2.imshow(str,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tutcode():
    #import & threshold image
    img = cv2.imread('coins.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = numpy.ones((3,3),numpy.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = numpy.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]

if __name__ == "__main__":
    main()
