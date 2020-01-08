#   Antonio Leonti
#   12.29.19
#   Convolutional neural network: takes custom dataset (datasets defined in the
#   train() call in main())


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

TRANSFORMS = transforms.Compose(
    #normalize to mean and standard deviation of .5
    [transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
)

def main():
    root = "/Users/antoniovleonti/Desktop/Research/"
    classes = ( "amp_radiarta/", "glob_menardii/", "glob_ruber",
                "nglob_dutertrei/", "tril_sacculifer/" )

    net = ConvNet()
    #if already trained
    if "trained_net" in os.listdir(root+"cnn/"):
        #load model
        net.load_state_dict(torch.load(root+"cnn/trained_net"))

    else: #else train a new one
        print("\"trained_net\" not found; training now.")
        net.train(  load_dir(root+"data/train/", 128),
                    load_dir(root+"data/vali/", 128), #data loaders
                    1, .001 #epochs, learning rate
        )
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
                transform = TRANSFORMS
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
        super(ConvNet, self).__init__()

        # Convolution -> ReLU -> Max Pooling
        self.ConvSeq = nn.Sequential(
            # convolutional layer
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1),
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

    #train the network
    def train(self, train_loader, val_loader, n_epochs, learning_rate):
        """trains the neural network
        """
        #optimizer and loss functions
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()
        #start the clock
        train_start = time()

        #Loop for n_epochs
        for epoch in range(n_epochs):
            #initialize variables
            total_train_loss = 0

            for i, data in enumerate(train_loader, 0):
                #Reset running loss and time
                running_loss = 0.0
                epoch_start = time()

                #Get inputs & wrap them in a Variable object
                inputs, labels = Variable(data[0]), Variable(data[1])

                #Set the parameter gradients to zero
                optimizer.zero_grad()

                #Forward pass, backward pass, optimize
                outputs = self(inputs)
                loss_size = loss(outputs, labels)
                loss_size.backward()
                optimizer.step()

                #Print statistics
                running_loss += loss_size.data
                total_train_loss += loss_size.data

                #Every 10th batch of an epoch
                if (i + 1) % (len(train_loader)//10 + 1) == 0:
                    #print some stats
                    print("Epoch {}, {:d}%\ttrain_loss: {:.5f}, took: {:.2f}s".format(epoch+1, int(100 * (i+1) / len(train_loader)), running_loss / len(train_loader), time() - epoch_start))
                    #^^^ "len(train_loader)" = # of image batches

            #At the end of the epoch, do a pass on the validation set
            self.test(val_loader)

        print("Training finished, took {:.2f}s".format(time() - train_start))

    def test(self, test_loader):

        #loss function & variable
        loss = nn.CrossEntropyLoss()
        total_loss = 0

        with torch.no_grad(): #dont compute gradients

            for inputs, labels in test_loader:
                #Wrap tensors in Variables
                inputs, labels = Variable(inputs), Variable(labels)

                #Forward pass
                outputs = self(inputs)
                loss_size = loss(outputs, labels)
                total_loss += loss_size.data

        print("Test loss = {:.2f}".format(total_loss / len(test_loader)))

if __name__ == '__main__': #do not touch
    main()
