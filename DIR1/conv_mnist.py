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
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.dense = nn.Sequential(
            nn.Linear(14*14*128, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.conv2(x)
        y1 = torch.reshape(y1, [x.size(0), -1])

        y2 = self.dense(y1)

        return y2


if __name__ == '__main__':
    transf_data = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, ], std=[0.5, ])]
    )
    train_data = datasets.MNIST(root="../mnist", train=True, transform=transf_data, download=True)
    test_data = datasets.MNIST(root="../mnist", train=False, transform=transf_data, download=True)

    batch_size = 100
    train_loader = data.DataLoader(train_data, batch_size=100, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True)

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

            if i % 100 == 0:
                a.append(i + epoch*(len(train_data) / batch_size))
                b.append(loss.item())
                plt.figure()
                plt.clf()
                plt.plot(a, b)
                plt.xlabel("BATCH")
                plt.ylabel("LOSS")
                plt.pause(0.001)

                print("epoch:{}, loss:{:.3f}".format(epoch, loss))

        torch.save(net.state_dict(), "./mnist_params.pth")
        # torch.save(net, "mnist.pth")

    net.eval()
    eval_loss = 0
    eval_acc = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        output = net(x)
        loss = loss_func(output, y)

        eval_loss = loss.item() * y.size(0)
        argmax = torch.argmax(output, 1)
        eval_acc += (argmax == y).sum().item()

    mean_loss = eval_loss / len(test_data)
    mean_acc = eval_acc / len(test_data)
    print("mean_loss:{}, mean_acc:{}".format(mean_loss, mean_acc))













