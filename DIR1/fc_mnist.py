import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=512, ),
            nn.BatchNorm1d(512),  # 在输出通道上做归一化
            nn.ReLU(inplace=True)  # 是否释放内存
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = torch.reshape(x, [x.size(0), -1])
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        self.y4 = self.fc4(y3)
        output = torch.softmax(self.y4, 1)
        return output


if __name__ == '__main__':
    transf_data = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, ], std=[0.5, ])]
    )
    batch_size = 100
    train_data = datasets.MNIST(root="../mnist", train=True, transform=transf_data, download=True)
    test_data = datasets.MNIST(root="../mnist", train=False, transform=transf_data, download=True)

    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # print(train_data.data.shape)
    # print(test_data.data.shape)
    # print(train_data.targets.shape)
    # print(test_data.targets.shape)
    # print(train_data.classes)
    # print(train_data.train)
    # print(test_data.classes)
    # print(test_data.train)

    # 在图片装在完成以后，选择其中一个批次的数据进行预览
    # images, labels = next(iter(train_loader))
    # img = make_grid(images)
    #
    # img = img.numpy().transpose(1, 2, 0)
    # std = [0.5, 0.5, 0.5]
    # mean = [0.5, 0.5, 0.5]
    # img = img*std + mean
    # print(labels)
    # print([labels[i] for i in range(100)])
    #
    # cv2.imshow("win", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Net().to(device)

    # net.load_state_dict(torch.load("./mnist_params.pth"))
    # net = torch.load("./mnist.net_pth")

    loss_func = nn.CrossEntropyLoss()  # 自带softmax, 自带one-hot
    optim = torch.optim.Adam(net.parameters())

    plt.ion()
    a = []
    b = []
    net.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            i = i + epoch*(len(train_data) / batch_size)
            x = x.to(device)
            y = y.to(device)

            out = net(x)
            loss = loss_func(net.y4, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 100 == 0:
                a.append(i)
                b.append(loss.item())
                plt.figure()
                plt.clf()
                plt.plot(a, b)
                plt.xlabel("BATCH")
                plt.ylabel("LOSS")
                plt.pause(1)

                print("Epoch:{}, loss:{:.3f}".format(epoch, loss.item()))

        torch.save(net.state_dict(), "./mnist_params.pth")
        # torch.save(net, "./mnist_net.pth")

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        loss = loss_func(out, y)

        eval_loss += loss.item() * y.size(0)
        arg_max = torch.argmax(out, 1)
        eval_acc += (arg_max == y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)
    print(mean_loss, mean_acc)



















