import torch
from torchsummary import summary


class CNNNet(torch.nn.Module):

    def __init__(self,n_classes):
        super().__init__()
        # 4 conv blocks / Flattens / linear / softmax

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels =1,
                out_channels= 16,
                kernel_size = 3,
                stride = 1,
                padding = 2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(16)

        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(32)

        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(64)

        )

        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels = 64,
                out_channels = 128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(128),
            torch.nn.Dropout(0.3)
        )

        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(128*5*4, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self,input_data):
        x1 = self.conv1(input_data)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x_flatten = self.flatten(x4)
        logits = self.linear(x_flatten)
        predictions = self.softmax(logits)
        return predictions





if __name__ == "__main__":
    n_class = 10
    cnn = CNNNet(n_classes=n_class)
    summary(cnn.cuda(),(1,64,44))