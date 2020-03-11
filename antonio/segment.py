#   Antonio Leonti
#   3.2.2020
#   Adaptive watershed segmentation - based on the

from volume_fill import volume_fill
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
    data = volume_fill(data)

    watershed(data, )


# src should be one image which has already been thresholded
def watershed(src, ridge):
    ### REQUIRED VALUES: ###
    # bwconncomp(images)
        # this could be achieved using cv2.findContours (like pw_crop)
    contour = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    # labelmatrix(bwconncomp)
        # what is this???
    # regionprops(bwconncomp, 'area')
        # just finding the area of each region, can be achieved with cv2.findContours still (LOOK INTO: cv2.contourArea(<contour>))
    area = []
    for i, c in enumerate(contour):
        area.append(cv2.contourArea(c))
    # job.minarea
        # from matlab code: "regions with voxels less than this number will not be segmented"
    # water.ridge
        # a bitmap image the same size as the slices
    # water.cutlarge
        # unknown variable??? used to test if the function has been run before?



if __name__ == "__main__":
    main()
