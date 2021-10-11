import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 超参数
BATCH_SIZE = 60
LR_G = 0.003
LR_D = 0.003
N_IDEAS = 5
EPOCH = 100
DOWNLOAD_MNIST = True

# cv2.namedWindow('show', cv2.WINDOW_NORMAL)
# plt.ion()

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.generator0 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator1 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator2 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator3 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator4 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator5 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator6 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator7 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator8 = nn.Linear(N_IDEAS, 28 * 28)
        self.generator9 = nn.Linear(N_IDEAS, 28 * 28)

    def forward(self, x, labels):
        g0 = self.generator0(x).unsqueeze(1)
        g1 = self.generator1(x).unsqueeze(1)
        g2 = self.generator2(x).unsqueeze(1)
        g3 = self.generator3(x).unsqueeze(1)
        g4 = self.generator4(x).unsqueeze(1)
        g5 = self.generator5(x).unsqueeze(1)
        g6 = self.generator6(x).unsqueeze(1)
        g7 = self.generator7(x).unsqueeze(1)
        g8 = self.generator8(x).unsqueeze(1)
        g9 = self.generator9(x).unsqueeze(1)

        y = torch.cat([g0, g1, g2, g3, g4, g5, g6, g7, g8, g9], dim=1)
        output = torch.zeros(60, 1, 784)
        for index, label in enumerate(labels):
            output[index, 0, :] = y[index, label, :]
        output = output.view((-1, 1, 28, 28))
        return output


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()

        self.detector0 = nn.Linear(28 * 28, 1)
        self.detector1 = nn.Linear(28 * 28, 1)
        self.detector2 = nn.Linear(28 * 28, 1)
        self.detector3 = nn.Linear(28 * 28, 1)
        self.detector4 = nn.Linear(28 * 28, 1)
        self.detector5 = nn.Linear(28 * 28, 1)
        self.detector6 = nn.Linear(28 * 28, 1)
        self.detector7 = nn.Linear(28 * 28, 1)
        self.detector8 = nn.Linear(28 * 28, 1)
        self.detector9 = nn.Linear(28 * 28, 1)

    def forward(self, x, label):
        x = x.view(x.size(0), -1)

        d0 = self.detector0(x)
        d1 = self.detector1(x)
        d2 = self.detector2(x)
        d3 = self.detector3(x)
        d4 = self.detector4(x)
        d5 = self.detector5(x)
        d6 = self.detector6(x)
        d7 = self.detector7(x)
        d8 = self.detector8(x)
        d9 = self.detector9(x)

        x = torch.cat([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9], dim=1).unsqueeze(1)

        label = torch.unsqueeze(label, 2)
        predict = torch.sigmoid(torch.bmm(x, label))
        return predict


D = D_Net()
G = G_Net()

print("D_net:", D)
print("G_net:", G)

opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

for epoch in range(EPOCH):
    for step, (data, target) in enumerate(train_loader):
        tar = torch.zeros((BATCH_SIZE, 10))
        target = torch.unsqueeze(target, 1)
        onehot_tar = tar.scatter_(1, target, 1)
        ideas = torch.randn((BATCH_SIZE, N_IDEAS))

        Fpaint = G(ideas, target)

        prob0 = D(Fpaint, onehot_tar)
        prob1 = D(data, onehot_tar)

        D_loss = - torch.mean(torch.log(prob1) + torch.log(1. - prob0))
        G_loss = torch.mean(torch.log(1. - prob0))
        if step < 500 and torch.mean(prob0) > 0.1:
            opt_D.zero_grad()
            D_loss.backward(retain_graph=True)
            opt_D.step()
        else:
            opt_G.zero_grad()
            G_loss.backward()
            opt_G.step()

        if step % 250 == 0:
            print("Epoch:{0}, Step:{1}, D_loss: {2:.4f} G_loss: {3:.4f}, prob0: {4:.4f}, prob1:{5:.4f}".format(
                epoch, step, D_loss.data.numpy(), G_loss, torch.mean(prob0).data.numpy(), torch.mean(prob1).data.numpy()))
            # plt.clf()
            plt.text(-.5, -1, '{0}'.format(target[0]))
            plt.imshow(Fpaint[0].view((28, 28)).detach().numpy(), cmap='gray')
            plt.show()
            # plt.draw()
            # plt.pause(0.01)
    # torch.save(G, "net.pkl")


