# Antonio Leonti
# 12.29.19
# Patchwise cropping is removing the empty space around an image's contents
# Uses: training and test data preprocessing


from os import listdir
from math import sqrt
from volume import show
import cv2
import numpy

exts = ("jpg", "jpeg", "png", "tif")

def main_segmented():
    root = "private/data/segmented/1/"
    for name in sorted(listdir(root)):
        # after checking its extension...
        if name.rpartition('.')[-1].lower() in exts:
            # open & crop image
            img = pw_crop(src = cv2.imread(root+name,0), thr = 0.1)
            show(img)


def main_test(): # preprocess a training & validation datset from a single source
    root = "data/"
    classes = ( # class names & thresholds
        ("amp_radiarta/", .01), ("glob_menardii/",.01), ("glob_ruber/", .05),
        ("nglob_dutertrei/",.05), ("tril_sacculifer/", .01)
    )
    l = 64 # height and width

    # for each species
    for c, t in classes:
        # get the names of each enclosed file
        for name in listdir(root+"src/"+c):
            # after checking its extension...
            if name.rpartition('.')[-1].lower() in exts:
                # determine output path (validation or training)
                path = "test/train/" if name.rpartition('_')[-1][0]!='5' else "test/vali/"
                # open & crop image
                img = pw_crop(src = cv2.imread(root+"src/"+c+name,0), thr = t)
                show(img);
                # resize & save
                cv2.imwrite(
                    root + path + c + name,
                    cv2.resize(img, (l,l))
                )


def pw_crop(src, thr = 0):
    """pw_crop() ("patchwise" crop) removes the black space around an image
    (src)'s contents. Parameter "thr" represents the fraction of the image
    which disjoint content must exceed (THReshold) to affect the crop.
    """
    #threshold, then find contours
    ctr = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)[1]
    ctr = cv2.findContours(ctr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    xmin, ymin = (max(src.shape[:2]), )*2 # initialize to image limits
    xmax, ymax = (0, )*2

    # combine bounding boxes
    for c in ctr:
        x, y, dx, dy = cv2.boundingRect(c) #candidate bounds
        # ensure current contour meets thresh
        if __hypot(dx, dy) > thr * __hypot(*src.shape[:2]):
            # resize bounding box if needed
            xmin, ymin = min(x, xmin), min(y, ymin)
            xmax, ymax = max(x + dx, xmax), max(y + dy, ymax)

    # find side length & new bottom left corner
    len = max(xmax - xmin, ymax - ymin)

    _xmin = (len - (xmax - xmin)) // 2
    _ymin = (len - (ymax - ymin)) // 2 # bring object to the center of the image
    _xmax = _xmin + xmax - xmin
    _ymax = _ymin + ymax - ymin

    # use numpy slicing to crop src, return result
    crop = numpy.zeros((len,len), dtype=src.dtype)
    crop[_ymin:_ymax,_xmin:_xmax] = src[ymin:ymax,xmin:xmax]

    return(crop)

def __hypot(a, b):
    """calculates hypotenuse of vectors a and b
    """
    return sqrt(a ** 2 + b ** 2)


if __name__ == "__main__":
    main_segmented()
