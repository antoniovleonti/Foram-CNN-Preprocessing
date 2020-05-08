# Antonio Leonti
# 3.7.2020
# Volume fill (volume_fill(3d_image)) aims to fill VOLUMES in a 3d image as opposed to filling regions in a 2D image.

import cv2
import numpy
import glob

def fill(data):

    # fill yz plane & transfer
    for x in range(data.shape[2]):
        # copy plane ("xz" because it's the x-z plane)
        yz = data[:,:,x].copy()
        # flood fill that plane
        cv2.floodFill(yz, None, (0, 0), 255)
        # invert and union with source data
        data[:,:,x] = data[:,:,x] | cv2.bitwise_not(yz)[:,:]

    # fill xz plane & transfer
    for y in range(data.shape[1]):
        xz = data[:,y,:].copy()
        cv2.floodFill(xz, None, (0, 0), 255)
        data[:,y,:] = data[:,y,:] | cv2.bitwise_not(xz)[:,:]

    # fill xy plane & transfer
    for z in range(data.shape[0]):
        xy = data[z,:,:].copy()
        cv2.floodFill(xy, None, (0, 0), 255)
        data[z,:,:] = data[z,:,:] | cv2.bitwise_not(xy)[:,:]

    return(data)

# example use
def main():
    root = "data/batch/"

    # get sorted list of files
    imgs = []
    for f in sorted(glob.glob(root + "*.tif")):
        # make a list of all (loaded) images
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

    # list -> numpy array
    data = numpy.array(imgs)

    # threshold images
    for z in range(data.shape[0]):
        data[z,:,:] = cv2.threshold(data[z,:,:], 0,255, cv2.THRESH_OTSU)[1]

    filled = fill(data)

    for z in range(filled.shape[0]):

        show(filled[z,:,:])


def show(img, str = "image"):
    cv2.imshow(str, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
