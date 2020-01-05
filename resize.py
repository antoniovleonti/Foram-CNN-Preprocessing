#   Antonio Leonti
#   12.29.19
#   Training dataset preprocessing


from os import listdir
from math import sqrt
import cv2
import numpy


def main():# example usage
    root = "/Users/antoniovleonti/Desktop/Research/"
    dirs=( # paths to images (relative to root/<src|data>/) & threshs
        ("validation/amphistegina/",.01), ("training/amphistegina/",.01),
        ("validation/glob/",.05),         ("training/glob/",.05)
    )
    l = 64 # height and width

    for d, t in dirs:
        # process, copy to another directory
        print("processing: "+root+"src/"+d)
        processdir(
            srcdir = root+"src/"+d,
            dstdir = root+"data/"+d,
            height = l, thresh = t
        )


def processdir(srcdir, height, dstdir = None, width = None, thresh = 0):
    """processdir() iterates through all images in srcdir/, 'frames' the
    contents (crop()'s in a square around it), then resizes the image to
    height x width before saving to dstdir/ (or back to srcdir/)
    """
    #default values of optional parameters
    if dstdir == None: dstdir = srcdir
    if width == None: width = height

    #get a list of all files in dir; iterate through them
    for name in listdir(srcdir):
        #check extension
        if name.rpartition('.')[-1].lower() in ("jpg", "jpeg", "png", "tif"):
            #crop
            img = crop(src = cv2.imread(srcdir+name,0), thresh = thresh)
            #resize & save
            cv2.imwrite(dstdir+name, cv2.resize(img, (width,height)))


def crop(src, thresh = 0):
    """crop() removes the empty space around an image's contents. It draws
    bounding boxes around
    """
    #threshold
    thr = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)[1]
    #findcontours
    ctr = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    xmin, ymin = (max(thr.shape[:2]), )*2 #initialize to opposite edges of image
    xmax, ymax = (0, )*2

    #make one big bounding box
    for c in ctr:
        x, y, dx, dy = cv2.boundingRect(c) #candidate coords
        # ensure current contour meets thresh
        if hypot(dx,dy) > thresh * hypot(*src.shape[:2]):
            #resize bounding box if needed
            xmin, ymin = min(x, xmin),min(y, ymin)
            xmax, ymax = max(x+dx, xmax), max(y+dy, ymax)

    #find side length & new bottom left corner
    len = max(xmax-xmin, ymax-ymin)
    xmin -= (len - (xmax-xmin))//2
    ymin -= (len - (ymax-ymin))//2

    #use numpy slicing to crop src
    return(src[ymin:ymin+len, xmin:xmin+len])

def hypot(a, b):
    """calculates hypotenuse of vectors a and b
    """
    return sqrt(a ** 2 + b ** 2)


if __name__ == "__main__":
    main()
