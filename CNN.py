import io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import requests
import gzip
from torch.utils.data import TensorDataset, DataLoader
import struct


def get_original_data():
    urls = [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ]
    filenames = [i.split("/")[-1] for i in urls]

    path = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(path, "DataSet")):
        os.mkdir(os.path.join(path, "DataSet"))
    path = os.path.join(path, "DataSet")

    for i in range(len(urls)):
        if not os.path.exists(os.path.join(path, filenames[i])):
            print("Downloading...")
            content = requests.get(urls[i]).content
            with open(os.path.join(path, filenames[i]), "wb") as f:
                f.write(content)

    with gzip.open(os.path.join(path, filenames[0]), "rb") as f:
        images_magic, images_num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            images_num, rows * cols
        )

    with gzip.open(os.path.join(path, filenames[1]), "rb") as f:
        labels_magic, labels_num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    with gzip.open(os.path.join(path, filenames[2]), "rb") as f:
        images_magic, images_num, rows, cols = struct.unpack(">IIII", f.read(16))
        images_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(
            images_num, rows * cols
        )

    with gzip.open(os.path.join(path, filenames[3]), "rb") as f:
        labels_magic, labels_num = struct.unpack(">II", f.read(8))
        labels_test = np.frombuffer(f.read(), dtype=np.uint8)

    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (images, labels, images_test, labels_test)
    )
    x_train, x_valid = map(
        lambda x: x.reshape(-1, 1, input_size, input_size).to(torch.float),
        [x_train, x_valid],
    )
    train_ds = TensorDataset(x_train.to(device), y_train.to(device))
    valid_ds = TensorDataset(x_valid.to(device), y_valid.to(device))
    return train_ds, valid_ds


def get_data(bs):
    train_ds, valid_ds = get_original_data()
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs),
    )


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(  # 输入(1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 进1个通道
                out_channels=16,  # 16个卷积核, 出16个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2,  # 边缘填充，若希望卷积后大小不变，则padding=(kenel_size-1)/2 if stride = 1
            ),  # 输出(16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 池化2x2, 输出(16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # 输入 (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 输出(32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出(32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def accuracy(predictions, labels):
    pred = torch.max(
        predictions, dim=1
    ).indices  # torch.max() 返回一个有名元组，元组内第一个为最大值们，第二个为最大值的索引们
    rights = pred.eq(
        labels.view_as(pred)
    ).sum()  # tensor1.view_as(tensor2) == tensor1.view(tensor2.shape)
    # rights是一维张量
    return rights, len(labels)


def fit():
    net = CNN().to(device)
    if os.path.exists(os.path.join(path, "model.pkl")):
        print("loading exist weight")
        net.load_state_dict(torch.load(os.path.join(path, "model.pkl"), map_location=device))
        net.eval()
        valid_rights = []
        for data, target in valid_dl:
            output = net(data)
            right = accuracy(output, target)  # right是元组
            valid_rights.append(right)
        valid_result = (
            sum([tup[0] for tup in valid_rights]),
            len(valid_dl.dataset),
        )
        print('测试集准确率为: {:.2f}%'.format(100 * valid_result[0] / valid_result[1]))
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    flag = 0
    for epoch in range(num_epoches):
        train_rights = []
        for batch_idx, (data, target) in enumerate(train_dl):
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_rights.append(right)
            if batch_idx % 100 == 99:
                net.eval()
                valid_rights = []
                for data, target in valid_dl:
                    output = net(data)
                    right = accuracy(output, target)  # right是元组
                    valid_rights.append(right)
                train_result = (
                    sum([tup[0] for tup in train_rights]),
                    sum([tup[1] for tup in train_rights]),
                )
                valid_result = (
                    sum([tup[0] for tup in valid_rights]),
                    len(valid_dl.dataset),
                )
                # (right_num, total_num)
                print(
                    "当前epoch{}\t[{}/{}({:.0f}%)]\t损失{:.6f}\t训练集准确率: {:.2f}%\t测试集准确率: {:.2f}%".format(
                        epoch,
                        batch_idx * batch_size,
                        len(train_dl.dataset),  # 数据集大小
                        100.0 * batch_idx / len(train_dl),  # batch数目
                        loss.data,
                        100.0 * train_result[0] / train_result[1],
                        100.0 * valid_result[0] / valid_result[1],
                    )
                )
                if valid_result[0] > flag:
                    torch.save(net.state_dict(), os.path.join(path, "model.pkl"))
                    flag = valid_result[0]
    


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.dirname(__file__)
    input_size, num_classes, num_epoches, batch_size = 28, 10, 3, 64
    train_dl, valid_dl = get_data(batch_size)
    fit()
    