# Antonio Leonti
# 12.29.19
# noise allows the user to generate an image with contents which are noise.

import os
from cv2 import imwrite
import numpy

def main(): # example usage
    #find avg images in each classifier
    home = "/Users/antoniovleonti/Desktop/Research/data/"
    sets = ("training/", "validation/")
    exts = ("jpg", "jpeg", "png", "tif")
    imgs = 0

    #create a noise classifier of the avg size of the other ones
    for set in sets: #dataset
        for __class in os.listdir(home+set): #classes
            if os.path.isdir(home+set+__class) and __class != "noise":
                for file in os.listdir(home+set+__class): #files (images)
                    #check extension
                    if file.rpartition('.')[-1].lower() in exts:
                        imgs+=1 #count

        #"len(...)-2"; 1 for .../.DS_STORE and another for .../noise/
        for i in range(imgs//(len(os.listdir(home+set))-2)):
            #write "that many" noise images
            imwrite(home+set+"noise/noise_"+str(i)+".png", noise(height=64))

        imgs = 0 #reset count for next dataset


def noise(height, width = None):
    """make_noise() returns a greyscale image (more precisely, a 1-D numpy
    array) of size height x width with randomized [0,255] contents
    """
    if width == None: width = height
    #an image is nothing more than a numpy array in opencv
    return(numpy.random.rand(height, width)*255)


if __name__ == "__main__":
    main()
