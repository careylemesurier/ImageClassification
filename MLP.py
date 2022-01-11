import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from keras.datasets import mnist


class DigitDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_sample = self.inputs[idx, :]
        label = self.labels[idx]
        return data_sample, label


class MLP(nn.Module):
    '''
    MLP Neural Netowrk Classifier
    '''

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 24),
            nn.ReLU(),
            nn.Linear(24, 10),
            nn.LogSoftmax()
        )

    def forward(self, inputs):
        return self.layers(inputs)


def train_MLP(mlp, trainloader, learning_rate, epochs):
    # Define loss and optimizing functions
    lossFun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        current_loss = 0
        for (batch_input, batch_labels) in trainloader:
            batch_input = batch_input.float()
            batch_labels = batch_labels.long()

            optimizer.zero_grad()

            # forward pass
            batch_pred = mlp.forward(batch_input)

            # Compute Loss
            loss = lossFun(batch_pred, batch_labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            current_loss = current_loss + loss.item()

        print(f'Epoch {epoch + 1}')
        print("average loss = ", current_loss / 10)

    return mlp


def test_MLP(mlp, testloader, batch_size):
    # add code
    confusion_matrix = np.zeros((10, 10))

    for (batch_input, batch_labels) in testloader:
        batch_input = batch_input.float()
        batch_labels = batch_labels.long()

        # forward pass
        batch_pred = mlp.forward(batch_input)

        # index of largest output = predicted digit
        pred_digits = torch.argmax(batch_pred, dim=1)

        for j in range(batch_size):
            confusion_matrix[pred_digits[j].item()][batch_labels[j].item()] += 1

    return confusion_matrix


def performance_metrics(confusion_matrix):
    TP = np.zeros(10)
    TN = np.zeros(10)
    FP = np.zeros(10)
    FN = np.zeros(10)

    size = np.sum(confusion_matrix)

    for digit in range(0, 10):
        TP[digit] = confusion_matrix[digit, digit]
        FP[digit] = np.sum(confusion_matrix[digit, :]) - TP[digit]
        FN[digit] = np.sum(confusion_matrix[:, digit]) - TP[digit]
        TN[digit] = size - TP[digit] - FP[digit] - FN[digit]

    precision = np.round(np.divide(TP, (TP + FP)), 4)
    recall = np.round(np.divide(TP, (TP + FN)), 4)
    f1_score = np.round(2*precision*recall/(precision+recall), 4)

    total_accuracy = np.sum(TP)/size

    return precision, recall, f1_score, total_accuracy


def main():
    # load data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # train_X = train_X.reshape(len(train_y), -1)
    # test_X = test_X.reshape(len(test_y), -1)

    train_X = train_X.reshape(len(train_y), 1, 28, 28)
    test_X = test_X.reshape(len(test_y), 1, 28, 28)

    # initiate data as a pytorch Dataset
    train_dataset = DigitDataset(train_X, train_y)
    test_dataset = DigitDataset(test_X, test_y)

    # load data as pytorch dataloader
    bs = 10
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True, num_workers=0)

    # initailize my MLP NN
    mlp = MLP()

    # train network
    learning_rate = 0.001
    epochs = 5
    mlp = train_MLP(mlp, trainloader, learning_rate, epochs)

    # test network
    cm = test_MLP(mlp, testloader, bs)
    precision, recall, f1_score, total_accuracy = performance_metrics(cm)
    print("test accuracy = ", total_accuracy)


if __name__ == '__main__':
    main()
