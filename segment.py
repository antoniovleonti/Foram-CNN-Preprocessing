#   Antonio Leonti
#   3.2.2020
#   Adaptive watershed segmentation - based on the

import volume
import edm
import cv2
import numpy
import glob

def main():
    root = "/Users/antoniovleonti/Documents/GitHub/foram-cnn/data/batch/"
    # load images
    imgs = []
    for f in sorted(glob.glob(root + "*.tif")):
        # make a list of all images
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
    # list -> numpy array
    data = numpy.array(imgs)

    for z in range(data.shape[0]):
        # threshold images
        data[z,:,:] = cv2.threshold(data[z,:,:], 0,255, cv2.THRESH_OTSU)[1]

    #fill volumes
    data = volume.fill(data)

    result = segment(data[0,:,:])

    show(result, "Watershed Result")


# src should be one image which has already been thresholded
def segment(img):

    minArea = 27000 # from the matlab code

    # find regions
    regions = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # for each region...
    for i, r in enumerate(regions):
        # get area
        area = cv2.contourArea(r)
        # draw the current shape
        canvas = numpy.zeros_like(img)
        cv2.drawContours(canvas, [r], 0, (0,255,0), 1)
        # flood fill inside of contour
        temp = canvas.copy()
        cv2.floodFill(canvas, None, (0, 0), 255)
        canvas = canvas | cv2.bitwise_not(temp)
        # create a euclidean distance map
        dist = cv2.distanceTransform(canvas, cv2.DIST_L2, 5)
        dist = img * cv2.GaussianBlur(dist, (5,5), cv2.BORDER_DEFAULT)

        # modify edm
        dist = edm.modify(-dist)

    return img



if __name__ == "__main__":
    test()
