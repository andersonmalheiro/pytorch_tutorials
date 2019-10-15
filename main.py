import torch
import torchvision
from core.net import ConvolutionalNN
from utils import imshow, loadData
import torch.optim as optim
import torch.nn as nn

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

MODEL_PATH = './models/cifar_net.pth'

# Loading train and test data
trainset, trainloader, testset, testloader = loadData()

# Getting random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Show images
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = ConvolutionalNN()

# Loss function and optimizer
# Cross-Entropy loss and Stochastic Gradient Descent
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training pass


def train(trainloader, n_epochs=2):
    print('Training...')
    for epoch in range(n_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs, data
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished training.')
    print('Saving model...')
    torch.save(net.state_dict(), MODEL_PATH)
    print('Model saved.')


train(trainloader)
