import cv2
import numpy
import glob

def main():
    root = "/Users/antoniovleonti/Documents/GitHub/foram-cnn/data/batch/"

    # get sorted list of files
    imgs = []
    for f in sorted(glob.glob(root + "*.tif")):
        # make a list of all images
        imgs.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))

    # list -> numpy array
    data = numpy.array(imgs)

    # threshold images
    for z in range(data.shape[0]):
        data[z,:,:] = cv2.threshold(data[z,:,:], 0,255, cv2.THRESH_OTSU)[1]

    for z in range(data.shape[0]):
        # fill from xz plane
        for y in range(data.shape[1]):
            # copy plane
            xz = data[:,y,:].copy()
            # flood fill that plane
            cv2.floodFill(xz, None, (0, 0), 255)
            # invert and union with original data
            data[z,y,:] = data[z,y,:] | cv2.bitwise_not(xz)[z,:]


    for z in range(data.shape[0]):

        show(data[z,:,:])


def show(img):
    cv2.imshow("3dfill.py",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
