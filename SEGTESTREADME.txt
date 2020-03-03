Watershed Base

Description
___________

This watershed algorithm segments scans of sediment into discrete objects. This segmentation is accomplished by thresholding the image to black/white, where black is the background and white is areas of interest (potential objects). The white pixels are then converted to grayscale based on the Euclidean distance to their nearest black pixel. The grayscale image is then inverted, resulting in grayscale "basins" where the outer edge is pure white, and the centers are progressively darker. These basins are then "filled" from darkest to lightest, or bottom to top (hence watershed). At the point where the "water" in two filling basins would meet, the objects are segmented.