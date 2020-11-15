import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils import data
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = nn.Sequential(
            nn.Linear(14 * 14 * 128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    transf_data = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, ], std=[0.5, ])]
    )
    train_data = datasets.MNIST(root="mnist", train=True,
                                transform=transf_data, download=True)
    test_data = datasets.MNIST(root="mnist", train=False,
                               transform=transf_data, download=False)
    train_loader = data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    net = Model().to(device)
    loss_func = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters())

    print(net)

    plt.ion()
    a = []
    b = []
    net.train()
    for epoch in range(2):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            out = net(x)
            loss = loss_func(out, y)

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

        torch.save(net.state_dict(), "mnist_params.pth")  # 保存网络参数
        # torch.save(net, "mnist.pth")  # 保存网络参数和模型

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
        print(y)
        print(arg_max)
        exit()

    mean_loss = eval_loss / len(test_data)  # 在全部训练完以后的总损失除以测试数据的个数
    mean_acc = eval_acc / len(test_data)  # 在全部训练完以后的预测正确的个数除以总个数
    print("平均损失：{0}, 平均精度：{1}".format(mean_loss, mean_acc))


















