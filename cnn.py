#   Antonio Leonti
#   12.29.19
#   Convolutional neural network: takes custom dataset (datasets defined in the
#   train() call in main())


#import numpy #handles very large arrays
from numpy import arange, int64, random

#import torchvision
from torchvision import datasets, transforms

import torch
from torch.autograd import Variable
from torch import nn

from time import time


def main():
    in_shape = (1,64)

    classes = ("amphistegina", "glob")

    net = ConvNet()

    net.train(
        #data loaders
        loaddir("/Users/antoniovleonti/Desktop/Research/data/training", 64),
        loaddir("/Users/antoniovleonti/Desktop/Research/data/validation", 64),
        #epochs, learning rate
        3, .01
    )



def loaddir(dir, batch_size):
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
                #normalize to mean and standard deviation of .5
                    [transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
                )
            ),
            #num of images in training partitions
            batch_size=batch_size,
            #max processes retrieving data from drive at a time
            num_workers=0,
            shuffle=True
        )
    )

# https://pytorch.org/docs/stable/nn.html?highlight=torch.nn#torch.nn.Module
# direct implementation questions here^^
class ConvNet(nn.Module):

    # Our batch shape for input x is (3, 32, 32)

    def __init__(self): # overrides default; initializes ConvNet object
        super(ConvNet, self).__init__()

        # Convolution -> ReLU -> Max Pooling
        self.ConvSeq = nn.Sequential(
            # convolutional layer
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1),
            #nn.Dropout2d(p=0.5),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #pooling is used to make the detection of features less sensitive to scale and orientation changes
        )
        #Fully Connected -> Fully Connected
        self.FCSeq = nn.Sequential(
            #torch.nn.Linear(in_features, out_features)
            nn.Linear(3136, 64),
            nn.Linear(64, 2)
        )


    #   "Defines the computation performed at every call."
    # ! "Although the recipe for forward pass needs to be defined within forward(), one should call the Module instance afterwards instead [...] since [the module instance] takes care of running the registered hooks while [forward()] silently ignores them."

    def forward(self, x): #overrides default; forward pass; do not call
        """defines our computational graph
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
        #optimizer and loss functions
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss = nn.CrossEntropyLoss()
        #start the clock
        train_start = time()

        #Loop for n_epochs
        for epoch in range(n_epochs):
            #start the clock
            epoch_start = time()

            #initialize variables
            running_loss = 0.0
            total_train_loss = 0

            for i, data in enumerate(train_loader, 0):
                #Get inputs
                inputs, labels = data

                #Wrap them in a Variable object
                inputs, labels = Variable(inputs), Variable(labels)

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

                    #Reset running loss and time
                    running_loss = 0.0
                    epoch_start = time()

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
