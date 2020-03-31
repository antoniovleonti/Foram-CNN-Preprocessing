#   Antonio Leonti
#   3.2.2020
#   Adaptive watershed segmentation - based on the

from volume import show
import cv2
import numpy

def main():
    arr = numpy.array( [[5,5,5,5,5,5,5],
                        [5,0,3,1,4,2,5],
                        [5,5,5,5,5,5,5]] ) + 1

    res = imhmin(arr, 3)

    print(res)

# takes negative edm as parameter
def modify(edm):
    # get abs() of every value in minima, then make a sorted array of all unique values in that minima
    depths = numpy.unique(numpy.abs(local_min(edm)))

    # if at least 2 basins found
    if len(depths[1:]) >= 2:
        # idk what the goal of this is but it's in the matlab code so:
        temp = [*depths[2:], 0] - depths[1:]

        # %fill all basins by f*H_0
        # MEDM = imhmin(EDM,aratio * max(temp));

        # MEDM(MEDM < -max(temp)) = -max(temp);


def imhmin(src, h):
    # TODO: speed up function by cropping image

    edm = src.copy()
    # might need this later?
    mask = numpy.zeros((edm.shape[0] + 2, edm.shape[1] + 2))
    # d is the domain / all values contained in the array
    d = numpy.unique(edm)

    # for the index of each local minima (sorted gtl)
    indices = numpy.nonzero(local_min(edm)) # get indices
    indices = numpy.dstack((indices[0], indices[1]))[0].tolist() # zip
    # sort based on the value of edm[] at that index
    indices.sort(key = lambda _: edm[_[0],_[1]], reverse = True)

    for (x,y) in indices:
        start = edm[x,y] # remember original value of minima

        # for each in a list of heights greater than the starting height
        for i in range(*numpy.where(d==edm[x,y])[0], d.shape[0]-1):
            # prevent exceeding target height
            step = start + h if (d[i+1] - start > h) else d[i+1]

            #-------------- WORKS UNTIL HERE --------------#

            # complete floodFill syntax:
            # cv2.floodFill(image, mask, seed, newVal[, loDiff[, upDiff[, flags]]]) → retval, rect

            # fill UPWARD onto image (and onto mask?)
            # cv2.floodFill(edm.astype(numpy.int8), None, (x,y), step, 0, -(step - d[i]), 4)
            cv2.floodFill(edm.astype(numpy.double), None, (y,x), float(step), 0, float(step-d[i]), 4)

            # fill DOWNWARD NOT onto image
            # have you overflowed?


def local_min(src):
    # TODO: speed up function by cropping image
    edm = src.copy() # ??? does python not use pass by value??

    # pad image
    padded = numpy.zeros((edm.shape[0] + 2, edm.shape[1] + 2))
    padded[1 : edm.shape[0]+1, 1 : edm.shape[1]+1] = edm.copy()
    # for each row
    for y in range(edm.shape[0]):
        # for each column
        for x in range(edm.shape[1]):
            # "kernel" looks like this: (X -> {0, 1})
            # { 1   1   1       X = 1 iff the underlying pixel...
            #   1   X   1           1. is not equal to zero
            #   1   1   1 }     &&  2. is the local minimum
            edm[y,x] = int(    edm[y,x] == padded[y:y+3, x:x+3].min()
                            and edm[y,x] != 0)

    return(edm)


if __name__ == "__main__":
    main()
