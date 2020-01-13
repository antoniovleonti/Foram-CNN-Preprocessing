#   Antonio Leonti
#   12.29.19
#   Convolutional neural network


import os
#import numpy #handles very large arrays
from numpy import arange, int64, random
import cv2
#import torchvision
from torchvision import datasets, transforms
import torch
from torch.autograd import Variable
from torch import nn

from time import time
from math import ceil

def main():
    #transformation applied to all images

    root = "/Users/antoniovleonti/Desktop/Research/"
    classes = ( "amp_radiarta/", "glob_menardii/", "glob_ruber",
                "nglob_dutertrei/", "tril_sacculifer/" )

    net = ConvNet()
    #if already trained
    if "trained_net" in os.listdir(root+"cnn/") and False:
        #load model
        net.load_state_dict(torch.load(root+"cnn/trained_net"))

    else:
        time_t = time()
        #optimizer and loss functions
        optimizer = torch.optim.Adam(net.parameters(), lr = .0001)
        criterion = nn.CrossEntropyLoss()
        #create data loaders
        loader_t = load_dir(root+"data/train/", 128)
        loader_v = load_dir(root+"data/validation/", 128)

        for epoch in range(5):
            time_e = time()
            #perform one epoch of training, record the loss
            loss_t = net.train( loader    = loader_t,
                                optimizer = optimizer,
                                criterion = criterion )

            #At the end of the epoch, do a pass on the validation set
            loss_v = net.validate(loader_v)

            print( "Epoch #{:d};  T.loss {:.3f};  V.loss {:.3f};  Time {:.3f}"
                   .format(epoch, loss_t, loss_v, time()-time_e)              )

        print("\nTraining finished! Time: {:.3f}s".format(time()-time_t))

        torch.save(net.state_dict(), root+"cnn/trained_net")


def load_dir(dir, batch_size):
    """returns a dataloader from dir with batch_size
    """
    return(
        #create a data loader
        torch.utils.data.DataLoader(
            #from the dataset held in this directory
            datasets.ImageFolder(
                root = dir,
                #turn it into a tensor and normalize images
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
                )
            ),
            #num of images in training partitions
            batch_size = batch_size,
            #max processes retrieving data from drive at a time
            num_workers = 2,
            shuffle = True
        )
    )

# https://pytorch.org/docs/stable/nn.html?highlight=torch.nn#torch.nn.Module
# direct most implementation questions here^^
class ConvNet(nn.Module):

    def __init__(self): # overrides default; initializes ConvNet object
        """Initializes all new ConvNet object as well as layers needed for our
        computational graph
        """
        super(ConvNet, self).__init__()

        # { Convolution -> ReLU -> Max Pool }x2
        self.ConvSeq = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        #Fully Connected -> Fully Connected
        self.FCSeq = nn.Sequential(
            #torch.nn.Linear(in_features, out_features)
            nn.Linear(3136, 64),
            nn.Linear(64, 5)
        )


    def forward(self, x): #overrides default; do not call
        """defines our computational graph. Do not call this method
        """
        #Conv -> Dropout -> ReLU -> Max Pool
        x = self.ConvSeq(x)
        #reshape tensor, doesn't copy memory
        x = x.view(-1, 3136)
        #fully connected -> fully connected
        x = self.FCSeq(x)
        return x


    def train(self, loader, optimizer, criterion = nn.CrossEntropyLoss()):
        """a single training epoch
        """
        lossSum = 0
        for i, (inputs, labels) in enumerate(loader):
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            #Forward pass, backward pass, optimize
            loss = criterion(self(inputs), labels) #self(inputs) = predictions
            loss.backward()
            optimizer.step()
            #record loss
            lossSum += loss.data
            #progress bar
            print(  "\r>Training ",
                    *('█' for _ in range((i+1) // ceil(len(loader)/10))),
                    *('▁' for _ in range(10 - (i+1) // ceil(len(loader)/10))),
                    "{:3d}% ".format(int(100 * (i/len(loader)))),
                    "Loss: {:5.3f}".format(lossSum / i),
                    sep = '', end = '\r', flush = True
            )

        return(float(lossSum / len(loader)))


    def validate(self, loader, criterion = nn.CrossEntropyLoss()):
        """ exactly the same as training, but without backpropagation
        """
        lossSum = 0
        with torch.no_grad(): #dont compute gradients (much faster)
            for inputs, labels in loader:
                #Forward pass
                loss = criterion(self(inputs), labels)
                lossSum += loss.data

        return(float(lossSum / len(loader)))

if __name__ == "__main__":
    main()
