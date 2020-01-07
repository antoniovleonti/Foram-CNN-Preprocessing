#   Antonio Leonti
#   12.29.19
#   Training dataset preprocessing


from os import listdir
from math import sqrt
import cv2
import numpy


def main(): # preprocess a training & validation datset from a single source
    root = "/Users/antoniovleonti/Desktop/Research/data/"
    classes = ( # paths to images (relative to root/<src|data>/) & threshs
        ("amp_radiarta/", .01), ("glob_menardii/",.01), ("glob_ruber/", .05),
        ("nglob_dutertrei/",.05), ("tril_sacculifer/", .01)
    )
    exts = ("jpg", "jpeg", "png", "tif")
    l = 64 # height and width

    # for each species
    for c, t in classes:
        # get the names of each enclosed file
        for name in listdir(root+"src/"+c):
            # check extension
            if name.rpartition('.')[-1].lower() in exts:
                # determine output path
                path = "train/" if name.rpartition('_')[-1][0]!='5' else "vali/"
                # open & crop image
                img = crop(src = cv2.imread(root+"src/"+c+name,0), thresh = t)
                # resize & save
                cv2.imwrite(
                    root + path + c + name,
                    cv2.resize(img, (l,l))
                )


def crop(src, thresh = 0):
    """crop() removes the empty space around an image's contents. For an object
    to expand the crop, its hypotenuse must be as big as the hypotenuse of src
    times thresh (if hypot(src)=100 and thresh=.05, hypot(obj) must be >= 5)
    """
    #threshold, then find contours
    ctr = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)[1]
    ctr = cv2.findContours(ctr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

    xmin, ymin = (max(src.shape[:2]), )*2 #initialize to opposite edges of image
    xmax, ymax = (0, )*2

    #make one big bounding box
    for c in ctr:
        x, y, dx, dy = cv2.boundingRect(c) #candidate bounds
        # ensure current contour meets thresh
        if hypot(dx, dy) > thresh * hypot(*src.shape[:2]):
            #resize bounding box if needed
            xmin, ymin = min(x, xmin), min(y, ymin)
            xmax, ymax = max(x + dx, xmax), max(y + dy, ymax)

    #find side length & new bottom left corner
    len = max(xmax - xmin, ymax - ymin)
    xmin -= (len - (xmax - xmin)) // 2
    ymin -= (len - (ymax - ymin)) // 2 # bring object to the center of the image

    #use numpy slicing to crop src, return result
    return(src[ymin:ymin+len, xmin:xmin+len])


def hypot(a, b):
    """calculates hypotenuse of vectors a and b
    """
    return sqrt(a ** 2 + b ** 2)


if __name__ == "__main__":
    main()
