# Sandy ConvNet
A ConvNet aimed at identifying various species of foraminifera in microCT scans of sand.
Utilizes PyTorch and NumPy

# UNIMPLEMENTED FEATURES

Training data

1. Synthesis (Drishti stuff)
2. Resize (32x?, 64x?)

Test data

1. Synthesis (Drishti stuff)
2. Segmentation
3. Grouping
4. Merging
5. Cropping

ConvNet

A separate driver program should probably be written to keep everything nice and modular (and create another layer of abstraction between "us" and the CNN).

1. Save trained model
2. Import training data
3. Import test data
4. Inference / classification of individual images
5. Infer classification of directories by that of their parts
6. Calculation of %volume or real volume occupied by target species
