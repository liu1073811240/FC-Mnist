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
            nn.Linear(in_features=784, out_features=512),
            nn.BatchNorm1d(512),  # 在输出通道上做归一化
            # nn.LayerNorm(512),
            # nn.InstanceNorm1d(512),
            # nn.GroupNorm(2, 512),

            nn.ReLU(inplace=False)  # inplace是否释放内存
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, x):  # 实例化网络时，call方法直接调用forward。
        x = torch.reshape(x, [x.size(0), -1])  # 将数据形状转成N,V结构 , V=C*H*W
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        y3 = self.fc3(y2)
        self.y4 = self.fc4(y3)
        output = torch.softmax(self.y4, 1)  # 输出（N, 10）.第0轴是批次，第一轴是数据. 作用：将实数值转为概率值
        return output


if __name__ == '__main__':
    transf_data = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, ], std=[0.5, ])]
    )
    train_data = datasets.MNIST(root="mnist", train=True,
                                transform=transf_data, download=True)  # 训练集
    test_data = datasets.MNIST(root="mnist", train=False,
                                transform=transf_data, download=False)  # 测试集
    batch_size = 100
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # print(train_data.data.shape)  # torch.Size([60000, 28, 28])
    # print(test_data.data.shape)  # torch.Size([10000, 28, 28])
    # print(train_data.targets.shape)  # torch.Size([60000])
    # print(test_data.data.shape)  # torch.Size([10000, 28, 28])
    # print(train_data.classes)  # 训练集的数据种类
    # print(train_data.train)  # 是否是参加训练的数据
    # print(test_data.classes)  # 测试集的数据种类
    # print(test_data.train)  # 是否是参加训练的数据

    # 在装载完成后，我们可以选取其中一个批次的数据进行预览
    # images, labels = next(iter(train_loader))
    # img = make_grid(images)  # (N, C, H, W) --> (C, H, W)
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
    # net = Net().to(device)

    # net.load_state_dict(torch.load("./mnist_params.pth"))  # 只恢复网络参数
    net = torch.load("./mnist.net_pth")  # 恢复保存的整个网络和参数

    loss_func = nn.CrossEntropyLoss()  # 自带softmax, 自带one-hot
    opt = torch.optim.Adam(net.parameters())

    plt.ion()
    a = []
    b = []
    net.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            i = i + epoch*(len(train_data)/batch_size)  # (0~599) + epoch*600   只适用于训练集长度能被100除断。
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_func(net.y4, y)  # 交叉熵损失函数自带softmax、one-hot
            # loss = loss_func(out, y)  # 两次softmax

            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:  # i和epoch有关系
                a.append(i)
                # print(a)
                b.append(loss.item())
                plt.figure()
                plt.clf()
                plt.plot(a, b)
                plt.xlabel("BATCH")
                plt.ylabel("LOSS")
                plt.pause(1)

                print("Epoch:{}, loss:{:.3f}".format(epoch, loss.item()))

        # torch.save(net.state_dict(), "mnist_params.pth")  # 保存网络参数
        torch.save(net, "mnist.net_pth")  # 保存网络参数和模型

    net.eval()  # 将网络模型固定，只传训练数据，不传测试数据
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        out = net(x)
        loss = loss_func(out, y)

        eval_loss += loss.item() * y.size(0)  # 一张图片的损失乘以一批图片为批量损失
        arg_max = torch.argmax(out, 1)  # 取最大值索引也就是网络所预测的数字。
        eval_acc += (arg_max == y).sum().item()  # 在100张图片中，拿预测的数字和标签的数字对应相等的个数

    mean_loss = eval_loss / len(test_data)  # 在全部测试完以后的总损失除以测试数据的个数
    mean_acc = eval_acc / len(test_data)  # 在全部测试完以后的预测正确的个数除以总个数
    print(mean_loss, mean_acc)







