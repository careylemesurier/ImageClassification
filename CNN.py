import MLP
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from keras.datasets import mnist


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # first convolution layer with relu activation and maxpooling layer
        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second convolution layer with relu activation and maxpooling layer
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # one fully connected layer
        self.fc1 = nn.Linear(800, 500)
        self.relu3 = nn.ReLU()

        # softmax classifier
        self.fc2 = nn.Linear(500, 10)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # first conv layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # second conv layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)

        # log soft max classification layer
        x = self.fc2(x)
        output = self.lsm(x)
        # return the output predictions
        return output


def main():
    # load data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # train_X = train_X.reshape(len(train_y), -1)
    # test_X = test_X.reshape(len(test_y), -1)

    train_X = train_X.reshape(len(train_y), 1, 28, 28)
    test_X = test_X.reshape(len(test_y), 1, 28, 28)

    # initiate data as a pytorch Dataset
    train_dataset = MLP.DigitDataset(train_X, train_y)
    test_dataset = MLP.DigitDataset(test_X, test_y)

    # load data as pytorch dataloader
    bs = 10
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True, num_workers=0)

    # initailize my CNN
    cnn = CNN()

    # train network
    learning_rate = 0.001
    epochs = 5
    cnn = MLP.train_MLP(cnn, trainloader, learning_rate, epochs)

    # test network
    cm = MLP.test_MLP(cnn, testloader, bs)
    precision, recall, f1_score, total_accuracy = MLP.performance_metrics(cm)
    print("test accuracy = ", total_accuracy)


if __name__ == '__main__':
    main()