# Antonio Leonti
# 3.31.2020
# reslice slices a 3d image from xz and yz planes (as opposed to normal xy plane)

import cv2
import numpy
import glob

def main():
    root = "data/batch/"

    # get sorted list of files
    imgs = []
    for f in sorted(glob.glob(root + "*.tif")):
        # make a list of all (loaded) images
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

    # list -> numpy array
    data = numpy.array(imgs)

    sliceXZ(data, root +"xz/xz_")


def sliceXZ(data, path):

    for x in range(data.shape[2]):
        cv2.imwrite(path+str(x)+".png", data[:,:,x])

def sliceYZ(data, path):

    for y in range(data.shape[1]):
        cv2.imwrite(path+str(y)+".png", data[:,y,:])


if __name__ == "__main__":
    main()
