#   Antonio Leonti
#   12.29.19
#   Test CNN: Made to familiarize myself with CNN architecture, python, and PyTorch.
#   MY BELOVED RESEARCH GROUP: PLEASE DON'T CHANGE WHAT YOU DON'T UNDERSTAND!:)



import numpy as np #handles very large arrays

from torchvision.datasets import CIFAR10 #our dataset
from torchvision import transforms

import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import Adam #optimizer
from torch import manual_seed
from torch import nn

from time import time



def main():
    #set a standard random seed for reproducible results.
    np.random.seed(144)
    manual_seed(144)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #download data
    train_set = CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

    #counts of images
    n_training = 20000
    n_val = 5000
    n_test = 5000

    #Make samplers for each
    train_sampler = SubsetRandomSampler(np.arange(n_training, dtype=np.int64))
    val_sampler = SubsetRandomSampler(np.arange(n_training, n_training + n_val, dtype=np.int64))
    test_sampler = SubsetRandomSampler(np.arange(n_test, dtype=np.int64))
    #numpy.arange(min,max) = min, min+1, ..., max-1

    #Make data loaders for each
    train_loader = DataLoader(train_set, batch_size=32, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)

    #We then designate the 10 possible labels for each image:
    classes = ( 'plane', 'car', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck' )

    net = ConvNet() #initialize net
    #train net
    net.train(train_loader, val_loader, 5, 0.001)
    #see how it performs on test data
    net.test(test_loader)



# https://pytorch.org/docs/stable/nn.html?highlight=torch.nn#torch.nn.Module
#direct implementation questions here^^
class ConvNet(nn.Module):

    #Our batch shape for input x is (3, 32, 32)

    def __init__(self): #overrides default; initializes ConvNet object

        super(ConvNet, self).__init__()

        #creating layers for later use; order doesn't matter

        #Convolution -> ReLU -> Max Pooling
        self.ConvSeq = nn.Sequential(
            #convolutional layer
            nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1),
            #nn.Dropout2d(p=0.5),
            #nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            #pooling is used to make the detection of features less sensitive to scale and orientation changes
        )
        #Fully Connected -> Fully Connected
        self.FCSeq = nn.Sequential(
            #torch.nn.Linear(in_features, out_features)
            nn.Linear(18*16*16, 64),
            nn.Linear(64, 10)
        )

    #   "Defines the computation performed at every call."
    # ! "Although the recipe for forward pass needs to be defined within forward(), one should call the Module instance afterwards instead [...] since [the module instance] takes care of running the registered hooks while [forward()] silently ignores them."

    def forward(self, x): #overrides default; forward pass; do not call
        #Conv -> Dropout -> ReLU -> Max Pool
        x = self.ConvSeq(x)
        #reshape tensor, doesn't copy memory
        x = x.view(-1, 18*16*16)
        #fully connected -> fully connected
        x = self.FCSeq(x)
        return x

    #train the network
    def train(self, train_loader, val_loader, n_epochs, learning_rate):
        #optimizer and loss functions
        optimizer = Adam(self.parameters(), lr=learning_rate)
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

                #Print every 10th batch of an epoch
                if (i + 1) % (len(train_loader)//10 + 1) == 0:
                    print("Epoch {}, {:d}%\ttrain_loss: {:.2f}, took: {:.2f}s".format(epoch+1, int(100 * (i+1) / len(train_loader)), running_loss / len(train_loader), time() - epoch_start))
                    #^^^ len(train_loader) = # of image batches

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
