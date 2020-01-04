#   Antonio Leonti
#   12.29.19
#   Training dataset preprocessing


from os import listdir
import cv2
import numpy


def main():

    dirs = (("images/training/","_dump/training/"),
            ("images/validation/","_dump/validation/"))
    len = 64

    for srcdir, dstdir in dirs:
        #preprocess these folders
        processdir(srcdir = srcdir, dstdir = dstdir, height = len)


def processdir(srcdir, height, dstdir = None, width = None):
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
            cropped = crop(cv2.imread(srcdir+name,0))
            #resize & save
            cv2.imwrite(dstdir+name, cv2.resize(cropped,(width,height)))


def crop(src):
    """crop() removes the empty space around an image's contents. IT DOES NOT
    ANALYZE CONTENTS... Any "noise" will be included in the crop.
    """
    #threshold
    thr = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)[1]
    #findcontours
    ctr = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    xmin, ymin = (max(thr.shape[:2]), )*2 #initialize to opposite edges of image
    xmax, ymax = (0, )*2

    #make one big bounding box
    for c in ctr:
        x, y, __x, __y = cv2.boundingRect(c) #candidate coords
        #find corners
        if()
        xmin, ymin = min(x, xmin), min(y, ymin)
        xmax, ymax = max(x+__x, xmax), max(y+__y, ymax)

    #find side length & new bottom left corner
    len = max(xmax-xmin, ymax-ymin)
    xmin -= (len - (xmax-xmin))//2
    ymin -= (len - (ymax-ymin))//2

    #use numpy slicing to crop src
    return(src[ymin:ymin+len, xmin:xmin+len])


if __name__ == '__main__':
    main()
